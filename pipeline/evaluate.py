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


def score(model, db: str, images: str, split: str, batch_size: int) -> dict:
    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score

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
    y_score = np.concatenate(scores)
    y_true = np.concatenate(targets)

    per = {}
    for i, name in enumerate(CLASSES):
        support = int(y_true[:, i].sum())
        if support == 0:
            per[name] = {"n": 0, "ap": None, "precision": None, "recall": None}
            continue
        pred = (y_score[:, i] >= THRESHOLD).astype(int)
        tp = int(((y_true[:, i] == 1) & (pred == 1)).sum())
        fp = int(((y_true[:, i] == 0) & (pred == 1)).sum())
        fn = int(((y_true[:, i] == 1) & (pred == 0)).sum())
        per[name] = {
            "n": support,
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
    fired = (y_score[:, plat] >= THRESHOLD).any(axis=1)
    fp_rate = float((is_neg & fired).sum() / max(is_neg.sum(), 1))

    return {"split": split, "images": int(len(ds)), "threshold": THRESHOLD,
            "micro_roc_auc": micro, "macro_ap": float(np.mean(aps)) if aps else None,
            "fp_rate_on_negatives": fp_rate, "per_class": per}


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
    r = score(model, args.db, args.images, args.split, args.batch_size)
    r["source"] = args.hub or args.weights
    show(r, f"{r['source']}")
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
