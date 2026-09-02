#!/usr/bin/env python
"""
Score a checkpoint against a manifest split, and diff two scorings.

Why this is separate from the notebook
--------------------------------------
Two models are only comparable if they were measured on the same images. The
notebook reports metrics for the model it just trained, on whatever the split
table said at that moment -- so a run from before the corpus grew cannot be
compared with one from after, even though both printed a "TEST AUROC".

This scores any checkpoint against the split as it stands now, so an old model
and a new one can be put on the same footing.

Preprocessing comes from processor/preprocessing.py, the same module the serving
worker uses, so an evaluation cannot silently measure something the deployed
model would never see.

    python evaluate.py --weights ./swin_s3_base_224-xblockm-timm --json baseline.json
    python evaluate.py --hub howdyaendra/swin_s3_base_224-xblockm-timm
    python evaluate.py --compare baseline.json retrained.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "processor"))

from manifest import CLASSES, default_db, default_images  # noqa: E402

THRESHOLD = 0.8   # matches processor/constants.py


def load_model(weights: str | None, hub: str | None, num_classes: int):
    import torch
    from timm import create_model

    model = create_model("swin_s3_base_224", num_classes=num_classes, pretrained=False)
    if hub:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        path = hf_hub_download(repo_id=hub, filename="model.safetensors")
        state = load_file(path)
    else:
        p = Path(weights)
        f = p / "model.safetensors" if p.is_dir() else p
        if not f.is_file():
            sys.exit(f"no model.safetensors at {f}")
        from safetensors.torch import load_file
        state = load_file(str(f))

    # A head-shape mismatch means the checkpoint was trained on a different class
    # list -- most likely the deployed 13-class model against this 12-class one.
    # torch raises on shape mismatches even with strict=False, so catch it and
    # say what actually happened rather than surfacing a tensor-shape traceback.
    head_w = state.get("head.fc.weight")
    if head_w is not None and head_w.shape[0] != num_classes:
        sys.exit(f"checkpoint has {head_w.shape[0]} classes, the manifest has "
                 f"{num_classes}. Scoring one against the other is meaningless -- "
                 f"this is probably the pre-rename 13-class model.")
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except RuntimeError as e:
        sys.exit(f"checkpoint does not fit this architecture:\n  {e}")
    if missing or unexpected:
        print(f"  note: {len(missing)} missing / {len(unexpected)} unexpected keys")
    model.eval()
    return model


def raw_scores(model, db: str, images: str, split: str, batch_size: int):
    """(y_true, y_score) over a split. Shared by scoring and threshold tuning so
    both see exactly the same numbers."""
    import numpy as np
    import torch

    from dataset import load_manifest_dataset
    from preprocessing import build_transform

    ds = load_manifest_dataset(db, images, split=split)
    tf = build_transform()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    scores, targets = [], []
    for i in range(0, len(ds), batch_size):
        rows = ds[i:i + batch_size]
        batch = torch.stack([tf(im.convert("RGB")) for im in rows["image"]]).to(device)
        with torch.no_grad():
            probs = model(batch).sigmoid().float().cpu().numpy()
        scores.append(probs)
        targets.append(np.asarray(rows["labels"], dtype=float))
    return np.concatenate(targets), np.concatenate(scores)


def score(model, db: str, images: str, split: str, batch_size: int,
          thresholds: dict | None = None) -> dict:
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true, y_score = raw_scores(model, db, images, split, batch_size)

    def thr(c):
        return (thresholds or {}).get(c, THRESHOLD)

    per = {}
    for i, name in enumerate(CLASSES):
        support = int(y_true[:, i].sum())
        if support == 0:
            per[name] = {"n": 0, "ap": None, "precision": None, "recall": None}
            continue
        pred = (y_score[:, i] >= thr(name)).astype(int)
        tp = int(((y_true[:, i] == 1) & (pred == 1)).sum())
        fp = int(((y_true[:, i] == 0) & (pred == 1)).sum())
        fn = int(((y_true[:, i] == 1) & (pred == 0)).sum())
        per[name] = {
            "n": support,
            "threshold": thr(name),
            "ap": float(average_precision_score(y_true[:, i], y_score[:, i])),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
        }

    aps = [v["ap"] for v in per.values() if v["ap"] is not None]
    try:
        micro = float(roc_auc_score(y_true.ravel(), y_score.ravel()))
    except ValueError:
        micro = None

    # The share of true negatives that fire on ANY platform class -- the closest
    # thing to a production false-positive rate this split can give.
    neg_i = CLASSES.index("negative")
    plat = [i for i in range(len(CLASSES)) if i != neg_i]
    is_neg = y_true[:, neg_i] == 1
    fired = np.zeros(len(y_score), dtype=bool)
    for i in plat:
        fired |= y_score[:, i] >= thr(CLASSES[i])
    fp_rate = float((is_neg & fired).sum() / max(is_neg.sum(), 1))

    return {"split": split, "images": int(len(y_true)), "threshold": THRESHOLD,
            "thresholds": {c: thr(c) for c in CLASSES},
            "micro_roc_auc": micro, "macro_ap": float(np.mean(aps)) if aps else None,
            "fp_rate_on_negatives": fp_rate, "per_class": per}


MIN_TO_TUNE = 5      # positives in val below which a class is left alone


def tune(model, db: str, images: str, batch_size: int, target_precision: float,
         min_support: int = MIN_TO_TUNE) -> dict:
    """Pick a per-class threshold on the VALIDATION split.

    Tuning on test would fit the operating point to the set used to judge the
    result, which is the same mistake as training on it. val exists for exactly
    this and has otherwise gone unused.

    For each class, take the LOWEST threshold whose precision still meets the
    target -- lowest because everything above it costs recall for no gain in
    precision, and recall is what a single global 0.8 has been destroying.

    A class with too few validation positives is left at the global threshold. A
    threshold fitted to three images is noise dressed as a decision.
    """
    import numpy as np

    y_true, y_score = raw_scores(model, db, images, "val", batch_size)
    grid = np.round(np.arange(0.05, 1.00, 0.01), 2)
    out, notes = {}, {}

    for i, name in enumerate(CLASSES):
        if name == "negative":
            continue
        pos = y_true[:, i]
        support = int(pos.sum())
        if support < min_support:
            notes[name] = f"only {support} in val, left at {THRESHOLD}"
            continue

        best = None
        for t in grid:
            pred = y_score[:, i] >= t
            tp = int((pred & (pos == 1)).sum())
            fp = int((pred & (pos == 0)).sum())
            fn = int((~pred & (pos == 1)).sum())
            if tp == 0:
                continue
            prec, rec = tp / (tp + fp), tp / (tp + fn)
            if prec >= target_precision:
                best = (float(t), prec, rec)
                break           # grid ascends, so the first hit is the lowest
        if best is None:
            notes[name] = f"cannot reach {target_precision:.0%} precision in val"
            continue
        out[name] = best[0]
        notes[name] = f"val precision {best[1]:.2f}, recall {best[2]:.2f}"
    return {"thresholds": out, "notes": notes, "target_precision": target_precision}


def confirm_on_test(t: dict, before: dict, after: dict,
                    target_precision: float) -> tuple[dict, dict]:
    """Keep only the tuned thresholds whose gain survives the test split.

    tune() fits on val, which is the right split to fit on -- but with val
    supports in the single digits, "lowest threshold meeting a precision target"
    fits noise as readily as signal, and a threshold written from that goes
    straight into the live labeller.

    A threshold is only worth writing if, measured on held-out test, it actually
    BUYS recall and still holds the precision it was fitted to reach. Both
    numbers are already computed for the summary table; this just refuses to
    write the ones that fail.

    Observed 2026-09-02 on the 12-class checkpoint: of six proposed thresholds
    this keeps one. twitter's 0.97 cost 0.29 recall on the highest-volume class
    and landed at 0.94 test precision -- below the 0.95 it was fitted for.
    facebook's 0.29 showed val precision 1.00 and test precision 0.50.
    """
    kept: dict[str, float] = {}
    dropped: dict[str, str] = {}
    for name, thr in t["thresholds"].items():
        b = before["per_class"].get(name)
        a = after["per_class"].get(name)
        if not b or not a or b["ap"] is None or a["ap"] is None:
            dropped[name] = "not measurable on test"
        elif a["recall"] <= b["recall"] + 1e-9:
            dropped[name] = (f"no recall gain on test "
                             f"({b['recall']:.2f} -> {a['recall']:.2f})")
        elif a["precision"] < target_precision:
            dropped[name] = (f"test precision {a['precision']:.2f} < target "
                             f"{target_precision:.2f}")
        else:
            kept[name] = thr
    return kept, dropped


def show_confirmation(kept: dict, dropped: dict) -> None:
    print(f"\nconfirmed against the test split: keeping {len(kept)} of "
          f"{len(kept) + len(dropped)} proposed")
    for name, thr in sorted(kept.items()):
        print(f"  keep  {name:<13}{thr:.2f}")
    for name, why in sorted(dropped.items()):
        print(f"  drop  {name:<13}{why}")
    if not kept:
        print("  nothing survived -- every class stays at the global threshold")


def show_tuning(t: dict, before: dict, after: dict) -> None:
    print(f"\ntuned on the val split for >= {t['target_precision']:.0%} precision")
    print(f"\n  {'class':<13}{'thr':>6}{'test recall':>14}{'test prec':>11}   note")
    for name in CLASSES:
        if name == "negative":
            continue
        thr = t["thresholds"].get(name, THRESHOLD)
        b, a = before["per_class"][name], after["per_class"][name]
        if b["ap"] is None:
            continue
        arrow = f"{b['recall']:.2f} -> {a['recall']:.2f}"
        mark = "  <-" if a["recall"] > b["recall"] + 1e-9 else ""
        print(f"  {name:<13}{thr:>6.2f}{arrow:>14}{a['precision']:>11.2f}"
              f"   {t['notes'].get(name, '')}{mark}")
    print(f"\n  macro AP is unchanged by thresholds -- AP is threshold-free.")
    print(f"  FP rate on negatives  {before['fp_rate_on_negatives']*100:.2f}% -> "
          f"{after['fp_rate_on_negatives']*100:.2f}%")


def show(r: dict, title: str = "") -> None:
    if title:
        print(f"\n{title}")
    print(f"  split={r['split']}  images={r['images']:,}  threshold={r['threshold']}")
    print(f"  micro ROC-AUC {r['micro_roc_auc']:.4f}   macro AP {r['macro_ap']:.4f}")
    print(f"  false positives on negatives: {r['fp_rate_on_negatives']*100:.2f}%")
    print(f"\n  {'class':<13}{'n':>6}{'AP':>8}{'prec':>8}{'recall':>8}")
    for name, v in r["per_class"].items():
        if v["ap"] is None:
            print(f"  {name:<13}{v['n']:>6}{'-':>8}{'-':>8}{'-':>8}")
        else:
            note = "  (n<8)" if v["n"] < 8 else ""
            print(f"  {name:<13}{v['n']:>6}{v['ap']:>8.3f}{v['precision']:>8.3f}"
                  f"{v['recall']:>8.3f}{note}")


def compare(a: dict, b: dict) -> None:
    if a["images"] != b["images"]:
        print(f"  WARNING: measured on different numbers of images "
              f"({a['images']:,} vs {b['images']:,}) -- not comparable")
    print(f"\n{'':<13}{'before':>10}{'after':>10}{'delta':>10}")
    for k, label in (("micro_roc_auc", "micro AUROC"), ("macro_ap", "macro AP")):
        d = b[k] - a[k]
        print(f"  {label:<13}{a[k]:>9.4f}{b[k]:>10.4f}{d:>+10.4f}")
    fa, fb = a["fp_rate_on_negatives"]*100, b["fp_rate_on_negatives"]*100
    print(f"  {'FP rate %':<13}{fa:>9.2f}{fb:>10.2f}{fb-fa:>+10.2f}"
          f"   {'better' if fb < fa else 'worse' if fb > fa else ''}")

    print(f"\n  {'class':<13}{'n':>5}{'AP before':>11}{'AP after':>10}{'delta':>9}"
          f"{'recall Δ':>10}")
    for name in a["per_class"]:
        va, vb = a["per_class"][name], b["per_class"].get(name, {})
        if va["ap"] is None or vb.get("ap") is None:
            continue
        d = vb["ap"] - va["ap"]
        rd = vb["recall"] - va["recall"]
        flag = "  <-" if abs(d) > 0.05 else ""
        print(f"  {name:<13}{vb['n']:>5}{va['ap']:>11.3f}{vb['ap']:>10.3f}"
              f"{d:>+9.3f}{rd:>+10.3f}{flag}")
    print("\n  A model is only worth shipping if macro AP is at least held AND the")
    print("  FP rate has not risen -- FP rate is what people actually report.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", help="local checkpoint dir or model.safetensors")
    p.add_argument("--hub", help="hub repo id, e.g. howdyaendra/…")
    p.add_argument("--db", default=default_db())
    p.add_argument("--images", default=default_images())
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--json", help="write the result here, for --compare later")
    p.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                   help="diff two saved results instead of scoring")
    p.add_argument("--tune-thresholds", action="store_true",
                   help="fit a per-class threshold on val, then report it on test")
    p.add_argument("--target-precision", type=float, default=0.95,
                   help="precision each tuned class must still reach (default 0.95)")
    p.add_argument("--write-constants", metavar="PATH",
                   help="write the tuned map as a python snippet for constants.py")
    args = p.parse_args()

    if args.compare:
        a = json.loads(Path(args.compare[0]).read_text())
        b = json.loads(Path(args.compare[1]).read_text())
        show(a, f"BEFORE  {args.compare[0]}")
        show(b, f"AFTER   {args.compare[1]}")
        compare(a, b)
        return

    if not (args.weights or args.hub):
        sys.exit("pass --weights or --hub (or --compare two saved results)")

    model = load_model(args.weights, args.hub, len(CLASSES))

    if args.tune_thresholds:
        t = tune(model, args.db, args.images, args.batch_size, args.target_precision)
        before = score(model, args.db, args.images, "test", args.batch_size)
        after = score(model, args.db, args.images, "test", args.batch_size,
                      thresholds=t["thresholds"])
        show_tuning(t, before, after)
        kept, dropped = confirm_on_test(t, before, after, args.target_precision)
        show_confirmation(kept, dropped)
        if args.write_constants:
            body = "\n".join(f'    "{k}": {v:.2f},'
                              for k, v in sorted(kept.items()))
            note = "".join(f"#   {n}: {w}\n" for n, w in sorted(dropped.items()))
            Path(args.write_constants).write_text(
                "# Fitted by evaluate.py --tune-thresholds on the val split,\n"
                f"# for >= {args.target_precision:.0%} precision per class, then\n"
                "# confirmed against test -- a threshold is written only if it\n"
                "# gains recall there AND holds its precision target.\n"
                "# Classes absent here fall back to THRESHOLD.\n"
                + (f"#\n# Proposed on val but dropped on test:\n{note}" if dropped else "")
                + f"CLASS_THRESHOLDS = {{\n{body}\n}}\n")
            print(f"\nwrote {args.write_constants} ({len(kept)} thresholds)")
        if args.json:
            Path(args.json).write_text(json.dumps(
                {"tuning": t, "test_before": before, "test_after": after}, indent=2) + "\n")
            print(f"wrote {args.json}")
        return

    r = score(model, args.db, args.images, args.split, args.batch_size)
    r["source"] = args.hub or args.weights
    show(r, f"{r['source']}")
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
