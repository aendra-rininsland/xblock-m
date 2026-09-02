#!/usr/bin/env python
"""
Fine-tune the classifier from the manifest. No notebook required.

This is a straight translation of xblock-m-timm.ipynb: same transforms, same
schedule, same loss, same seed. It is deliberately not an improvement on it --
the point is that a run from here can be compared with a run from there, so the
notebook can be retired without losing the thread.

What it gains over the notebook:

  * hyperparameters are a recorded command rather than cell state, so "was that
    run comparable?" has an answer
  * runs over SSH with no browser, and survives a dropped connection under nohup
  * writes a run.json beside the weights: the arguments, the class counts, the
    per-epoch curve. A checkpoint stops being an anonymous directory.

Preprocessing comes from processor/preprocessing.py, the same module the serving
worker uses, so training cannot drift from inference.

    python train.py --out runs/2026-09-02
    python train.py --out runs/exp-pos-weight --pos-weight --epochs 8
    nohup python train.py --out runs/overnight > runs/overnight.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "processor"))

from manifest import CLASSES, default_db, default_images  # noqa: E402

ARCH = "swin_s3_base_224"


def build_loaders(db, images, batch_size, seed, augment):
    import torch
    import torchvision.transforms as T

    from dataset import load_manifest_dataset
    from preprocessing import IMG_SIZE, NORM_MEAN, NORM_STD, build_transform

    # Screenshots are not natural images. No horizontal flip -- it mirrors text
    # and moves nav chrome to a side it never appears on. Rotation stays small;
    # at +/-30 degrees the corners also fill with wedges that never occur at
    # inference. Resize and Normalize must match preprocessing.py exactly.
    train_tf = T.Compose([
        T.Resize(IMG_SIZE),
        *( [T.RandomRotation(3)] if augment else [] ),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    valid_tf = build_transform()

    def make(split, tf, shuffle):
        ds = load_manifest_dataset(db, images, split=split)

        def apply(batch):
            batch["pixel_values"] = [tf(x.convert("RGB")) for x in batch["image"]]
            batch["labels"] = [torch.tensor(l, dtype=torch.float) for l in batch["labels"]]
            return batch

        def collate(b):
            return {"pixel_values": torch.stack([x["pixel_values"] for x in b]),
                    "labels": torch.stack([x["labels"] for x in b]).float()}

        return ds, torch.utils.data.DataLoader(
            ds.with_transform(apply), batch_size=batch_size, shuffle=shuffle,
            collate_fn=collate)

    tr_ds, tr = make("train", train_tf, True)
    _, va = make("val", valid_tf, False)
    return tr_ds, tr, va


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=default_db())
    p.add_argument("--images", default=default_images())
    p.add_argument("--out", required=True, help="directory for weights and run.json")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--warm-start", metavar="DIR_OR_HUB",
                   help="initialise the backbone from an existing checkpoint")
    p.add_argument("--pos-weight", action="store_true",
                   help="weight positives by their inverse frequency in BCE")
    p.add_argument("--no-augment", action="store_true")
    args = p.parse_args()

    import torch
    import torch.nn as nn
    from timm import create_model
    from transformers.optimization import get_cosine_schedule_with_warmup

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tr_ds, train_dl, valid_dl = build_loaders(
        args.db, args.images, args.batch_size, args.seed, not args.no_augment)
    print(f"device {device}   train {len(train_dl.dataset):,}   "
          f"val {len(valid_dl.dataset):,}   classes {len(CLASSES)}")

    counts = [0] * len(CLASSES)
    for row in tr_ds:
        for i, v in enumerate(row["labels"]):
            counts[i] += int(v)
    print("  " + "  ".join(f"{c}={n}" for c, n in zip(CLASSES, counts)))

    model = create_model(ARCH, num_classes=len(CLASSES), pretrained=not args.warm_start)
    if args.warm_start:
        from safetensors.torch import load_file
        src = Path(args.warm_start)
        if src.is_dir() or src.is_file():
            f = src / "model.safetensors" if src.is_dir() else src
            state = load_file(str(f))
        else:
            from huggingface_hub import hf_hub_download
            state = load_file(hf_hub_download(repo_id=args.warm_start,
                                              filename="model.safetensors"))
        # Drop the head: a warm start is about the backbone, and the head may
        # have been trained on a different class list entirely.
        state = {k: v for k, v in state.items() if not k.startswith("head.")}
        missing, _ = model.load_state_dict(state, strict=False)
        print(f"  warm-started from {args.warm_start} "
              f"({len(missing)} head params left random)")
    model.to(device)

    pos_weight = None
    if args.pos_weight:
        total = len(tr_ds)
        # Inverse frequency, clamped: ngl at 4 examples in 5,000 would otherwise
        # get a weight in the hundreds and dominate the gradient.
        pos_weight = torch.tensor(
            [min((total - n) / max(n, 1), 20.0) for n in counts],
            dtype=torch.float, device=device)
        print("  pos_weight " + "  ".join(f"{c}={w:.1f}"
              for c, w in zip(CLASSES, pos_weight.tolist())))

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    # Span the WHOLE run. Sizing this to one epoch is what made the original
    # schedule cycle back to full learning rate instead of decaying once.
    total_steps = len(train_dl) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps)

    history, started = [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_dl:
            logits = model(batch["pixel_values"].to(device))
            loss = loss_fn(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
        train_loss = running / len(train_dl)

        model.eval()
        running = 0.0
        with torch.no_grad():
            for batch in valid_dl:
                logits = model(batch["pixel_values"].to(device))
                running += loss_fn(logits, batch["labels"].to(device)).item()
        valid_loss = running / len(valid_dl)

        lr = scheduler.get_last_lr()[0]
        print(f"epoch {epoch}  lr {lr:.2e}  train_loss {train_loss:.4f}  "
              f"valid_loss {valid_loss:.4f}")
        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss,
                        "valid_loss": valid_loss})

        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in model.state_dict().items()},
                  str(out / "model.safetensors"))

    (out / "run.json").write_text(json.dumps({
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args), "classes": CLASSES, "train_class_counts":
        dict(zip(CLASSES, counts)), "train_images": len(tr_ds),
        "history": history, "minutes": round((time.time() - started) / 60, 1),
    }, indent=2) + "\n")

    print(f"\nwrote {out}/model.safetensors and run.json "
          f"({(time.time()-started)/60:.1f} min)")
    print("Metrics deliberately not reported here -- score it against the frozen "
          "split so\nit is comparable with every other model:")
    print(f"  python evaluate.py --weights {out} --json {out.name}.json")


if __name__ == "__main__":
    main()
