#!/usr/bin/env python
"""
Turn the manifest into multi-label training data.

The notebook currently builds its targets like this:

    labels = torch.tensor([[x] for x in batch['label']])
    batch['labels'] = nn.functional.one_hot(labels, num_classes).sum(dim=1)

which assumes exactly one integer label per image and produces a target vector
with a single 1. The head is sigmoid and the loss is BCEWithLogitsLoss, so the
architecture permits multi-label -- but that construction cannot express one, so
the nested-screenshot case is unlearnable however the model is trained.

This module reads the manifest, where labels are a set, and produces multi-hot
targets. Swap the notebook's data source for:

    from dataset import load_manifest_dataset
    ds = load_manifest_dataset(DB, IMAGES, split="train")

and drop the one_hot block -- `labels` arrives already multi-hot.

Reporting
---------
    python dataset.py --co-occurrence

reports multi-label COVERAGE: how many examples of each class co-occur with
another. Multi-label is a design property of the system, not something the data
has to justify -- but a class with no co-occurrence examples still cannot learn
to co-occur, so this is how you see which classes are covered and which are not.
"""
from __future__ import annotations

import argparse
import sqlite3
from collections import Counter
from pathlib import Path

from manifest import CLASSES, connect, image_path

HUMAN = "human"
IMPORTED_SOURCE = "imagefolder-v1"


def multi_hot(labels: list[str], classes: list[str] = CLASSES) -> list[float]:
    index = {c: i for i, c in enumerate(classes)}
    vec = [0.0] * len(classes)
    for label in labels:
        if label in index:
            vec[index[label]] = 1.0
    return vec


def read_rows(db_path: str, images_root: Path, split: str | None = None,
              sources: list[str] | None = None,
              classes: list[str] = CLASSES) -> list[dict]:
    """Manifest rows with their label sets, as plain dicts."""
    conn = connect(db_path)
    conn.row_factory = sqlite3.Row

    where, params = ["EXISTS (SELECT 1 FROM labels l WHERE l.cid = i.cid)"], []
    if split:
        where.append("i.cid IN (SELECT cid FROM splits WHERE split = ?)")
        params.append(split)

    rows = []
    for r in conn.execute(
        f"SELECT cid, path FROM images i WHERE {' AND '.join(where)} ORDER BY cid",
        params,
    ).fetchall():
        q = "SELECT label, source FROM labels WHERE cid = ?"
        qp: list = [r["cid"]]
        if sources:
            q += f" AND source IN ({','.join('?' * len(sources))})"
            qp += sources
        pairs = conn.execute(q, qp).fetchall()
        if not pairs:
            continue
        # A human decision supersedes an imported one. Without this an image
        # reviewed down to a single correct label would still carry the stale
        # ImageFolder label alongside it.
        human = [p["label"] for p in pairs if p["source"] == HUMAN]
        labels = sorted(set(human)) if human else sorted({p["label"] for p in pairs})
        rows.append({
            "cid": r["cid"],
            "image_path": str(image_path(images_root, r["cid"])),
            "labels": labels,
            "target": multi_hot(labels, classes),
            "reviewed": bool(human),
        })
    conn.close()
    return rows


def load_manifest_dataset(db_path: str, images_root: str, split: str | None = None,
                          sources: list[str] | None = None,
                          classes: list[str] = CLASSES):
    """A datasets.Dataset with `image`, `labels` (multi-hot) and `cid`.

    Ready for .with_transform() exactly like the ImageFolder dataset it replaces,
    except that `labels` is already a multi-hot vector.
    """
    import datasets

    rows = read_rows(db_path, Path(images_root), split, sources, classes)
    if not rows:
        raise ValueError(f"no labelled images in {db_path}"
                         + (f" for split={split!r}" if split else ""))

    ds = datasets.Dataset.from_dict({
        "cid": [r["cid"] for r in rows],
        "image": [r["image_path"] for r in rows],
        "labels": [r["target"] for r in rows],
    })
    return ds.cast_column("image", datasets.Image())


def label_stats(db_path: str, images_root: Path,
                classes: list[str] = CLASSES) -> dict:
    rows = read_rows(db_path, images_root, classes=classes)
    per_class = Counter(l for r in rows for l in r["labels"])
    sizes = Counter(len(r["labels"]) for r in rows)
    reviewed = [r for r in rows if r["reviewed"]]
    reviewed_multi = [r for r in reviewed if len(r["labels"]) > 1]
    pairs = Counter(
        tuple(sorted((a, b)))
        for r in rows if len(r["labels"]) > 1
        for i, a in enumerate(r["labels"]) for b in r["labels"][i + 1:]
    )
    return {"rows": rows, "per_class": per_class, "sizes": sizes,
            "reviewed": reviewed, "reviewed_multi": reviewed_multi, "pairs": pairs}


def main() -> None:
    here = Path(__file__).parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=str(here / "data" / "pipeline.db"))
    p.add_argument("--images", default=str(here / "data" / "images"))
    p.add_argument("--co-occurrence", action="store_true",
                   help="report how often an image carries more than one label")
    args = p.parse_args()

    s = label_stats(args.db, Path(args.images))
    rows = s["rows"]
    if not rows:
        print(f"{args.db} has no labelled images yet.")
        return

    print(f"{len(rows):,} labelled images\n")
    print("per class")
    for name in CLASSES:
        n = s["per_class"].get(name, 0)
        print(f"  {name:<14}{n:>7,}")

    print("\nlabels per image")
    for size in sorted(s["sizes"]):
        n = s["sizes"][size]
        print(f"  {size} label{'s' if size != 1 else ''}   {n:>7,}  ({n/len(rows)*100:>5.1f}%)")

    if args.co_occurrence:
        rev, rev_multi = s["reviewed"], s["reviewed_multi"]
        multi_all = [r for r in rows if len(r["labels"]) > 1]
        print("\n── multi-label coverage ──────────────────────────────────────")
        print(f"  images with >1 label   {len(multi_all):>7,}  of {len(rows):,}")
        print(f"  human-reviewed         {len(rev):>7,}")
        print(f"  ...with >1 label       {len(rev_multi):>7,}")

        if s["pairs"]:
            print("\n  observed pairs")
            for (a, b), n in s["pairs"].most_common(15):
                print(f"    {a} + {b:<18}{n:>6,}")

        # A class the model never sees co-occurring cannot learn to co-occur, so
        # the useful view is per class, not one headline percentage.
        covered = {c for pair in s["pairs"] for c in pair}
        missing = [c for c in CLASSES if c != "negative"
                   and s["per_class"].get(c, 0) and c not in covered]
        print("\n  classes with NO co-occurrence example yet:")
        print("    " + (", ".join(missing) if missing else "(none)"))
        if missing:
            print("\n  These can only ever be predicted alone. The multi-candidate")
            print("  harvest bucket targets exactly this gap.")


if __name__ == "__main__":
    main()
