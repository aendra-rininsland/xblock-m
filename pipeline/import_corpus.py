#!/usr/bin/env python
"""
Import the existing ImageFolder corpus into the multi-label manifest.

Applies the same class surgery as the training notebook -- drop `news`, rename
`altright` -> `truthsocial` -- so the two cannot disagree about what the class
list is.

The important caveat
--------------------
The source corpus is an ImageFolder tree: one directory per class, so exactly one
label per image. Those labels are therefore **"at least this label", not "exactly
this label"**. An image filed under twitter/ may well also be a Bluesky
screenshot; the old structure simply gave nobody anywhere to record that.

This matters under BCEWithLogitsLoss, where an unrecorded positive becomes an
explicit zero in the target -- actively training the model to suppress the
co-occurrence you are trying to teach it. So imported rows are written with
source='imagefolder-v1' rather than 'human', and with review_state='imported'
rather than 'done', which keeps "nobody has looked at this under the multi-label
regime" distinguishable from "a human confirmed only one label applies".

Re-review them in the UI with the `imported` bucket filter. `dataset.py
--co-occurrence` reports how often a second label gets added, which is the
cheapest way to find out whether nested screenshots are common enough to matter.

    python import_corpus.py --dry-run
    python import_corpus.py --apply
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

from manifest import CLASSES, connect, image_path

HF_DATASET = "howdyaendra/xblock-social-screenshots"
SOURCE = "imagefolder-v1"
IMPORTED = "imported"

# Kept identical to cell 11 of xblock-notebooks/xblock-m-timm.ipynb.
DROP_CLASSES = {"news"}
RENAME_CLASSES = {"altright": "truthsocial"}


def resolve_classes(original: list[str]) -> tuple[list[str], dict[int, str]]:
    """Return (kept class names, original index -> final class name).

    Separated from any I/O so the remap can be verified on its own -- getting it
    wrong would silently relabel the entire corpus.
    """
    kept = [RENAME_CLASSES.get(n, n) for n in original if n not in DROP_CLASSES]
    mapping = {
        i: RENAME_CLASSES.get(n, n)
        for i, n in enumerate(original)
        if n not in DROP_CLASSES
    }
    return kept, mapping


def cid_from_filename(path: str) -> str | None:
    """The corpus is named by blob CID (`altright/bafkrei....jpeg`), so imported
    images share an identity with harvested ones and deduplicate against them."""
    stem = Path(path).stem
    return stem if stem.startswith("bafkrei") and stem.isalnum() else None


def main() -> None:
    here = Path(__file__).parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=str(here / "data" / "pipeline.db"))
    p.add_argument("--images", default=str(here / "data" / "images"))
    p.add_argument("--dataset", default=HF_DATASET)
    p.add_argument("--apply", action="store_true", help="write (default is a dry run)")
    args = p.parse_args()

    try:
        import datasets
    except ImportError:
        sys.exit("pip install datasets")

    print(f"loading {args.dataset} ...")
    ds = datasets.load_dataset(args.dataset, split="train")
    # decode=False keeps the original encoded bytes. Re-encoding through PIL
    # would change the bytes and make the filename CID a lie.
    ds = ds.cast_column("image", datasets.Image(decode=False))

    original = ds.features["label"].names
    kept, mapping = resolve_classes(original)
    unknown = set(kept) - set(CLASSES)
    if unknown:
        sys.exit(f"corpus has classes the manifest does not know: {sorted(unknown)}")

    print(f"classes  {len(original)} -> {len(kept)}  (dropped {sorted(DROP_CLASSES)}, "
          f"renamed {RENAME_CLASSES})")

    images_root = Path(args.images)
    conn = connect(args.db)
    existing = {r[0] for r in conn.execute("SELECT cid FROM images")}

    now = datetime.now(timezone.utc).isoformat()
    counts: dict[str, int] = {}
    imported = skipped_dropped = skipped_existing = skipped_nocid = 0

    for row in ds:
        label_name = mapping.get(row["label"])
        if label_name is None:
            skipped_dropped += 1
            continue

        blob = row["image"]
        raw, path = blob["bytes"], blob.get("path") or ""
        cid = cid_from_filename(path)
        if cid is None:
            # Fall back to hashing so a renamed file is still importable, but it
            # will not deduplicate against harvested copies.
            if not raw:
                skipped_nocid += 1
                continue
            cid = "sha256" + hashlib.sha256(raw).hexdigest()[:40]

        if cid in existing:
            skipped_existing += 1
            continue

        if args.apply:
            dest = image_path(images_root, cid)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if raw:
                dest.write_bytes(raw)
            conn.execute(
                "INSERT OR IGNORE INTO images (cid, did, rkey, post_uri, path, bytes,"
                " harvested_at, bucket) VALUES (?,?,?,?,?,?,?,?)",
                (cid, "", "", "", str(dest.relative_to(images_root)),
                 len(raw or b""), now, IMPORTED))
            conn.execute(
                "INSERT OR REPLACE INTO labels (cid,label,source,created_at) VALUES (?,?,?,?)",
                (cid, label_name, SOURCE, now))
            conn.execute(
                "INSERT INTO review_state (cid,state,updated_at) VALUES (?,?,?) "
                "ON CONFLICT(cid) DO UPDATE SET state=excluded.state",
                (cid, IMPORTED, now))

        existing.add(cid)
        counts[label_name] = counts.get(label_name, 0) + 1
        imported += 1

    if args.apply:
        conn.commit()
    conn.close()

    verb = "imported" if args.apply else "would import"
    print(f"\n{verb} {imported:,} images")
    for name in sorted(counts, key=lambda n: -counts[n]):
        print(f"  {name:<14}{counts[name]:>6,}")
    print(f"\nskipped: {skipped_dropped:,} dropped-class, "
          f"{skipped_existing:,} already present, {skipped_nocid:,} unidentifiable")
    if not args.apply:
        print("\ndry run -- re-run with --apply to write")
    else:
        print(f"\nlabels written with source='{SOURCE}', review_state='{IMPORTED}'.")
        print("These are 'at least' labels. Re-review with the `imported` bucket "
              "filter, then `python dataset.py --co-occurrence` to see how often a "
              "second label gets added.")


if __name__ == "__main__":
    main()
