#!/usr/bin/env python
"""
Import the existing ImageFolder corpus into the multi-label manifest.

Applies the same class surgery as the training notebook -- drop `news`, rename
`altright` -> `truthsocial` -- so the two cannot disagree about what the class
list is.

Duplicate CIDs are multi-label
------------------------------
The corpus is an ImageFolder tree, but a CID appearing in two folders means the
image carries both labels -- so it does encode some co-occurrence, and every file
must be read rather than deduplicated on first sight. 37 of 1,581 distinct images
are filed twice.

Only 6 of those are genuine co-occurrence (0.38%): bluesky+twitter,
facebook+twitter, instagram+threads and so on. The other 31 pair a platform with
`negative`, which is a contradiction -- negative means "not a screenshot", so it
cannot hold alongside "is a screenshot of Discord". 29 of them are
discord+negative specifically, which looks like one bulk misfile rather than 29
independent judgements. Those are imported with NO label and queued for review
rather than resolved by guessing which folder was wrong.

The remaining caveat
--------------------
For the other 99.6%, a single label is still **"at least this label", not
"exactly this label"**. 0.38% is the rate at which somebody went to the trouble of
filing an image twice, which is a lower bound on true co-occurrence, not a
measurement of it: a nested screenshot filed once under its dominant platform is
indistinguishable from a single-platform one.

This matters under BCEWithLogitsLoss, where an unrecorded positive becomes an
explicit zero in the target -- actively training the model to suppress the
co-occurrence you are trying to teach it. So imported rows are written with
source='imagefolder-v1' rather than 'human', and with review_state='imported'
rather than 'done', which keeps "nobody has looked at this under the multi-label
regime" distinguishable from "a human confirmed only one label applies".

Re-review them in the UI with the `imported` bucket filter. `dataset.py
--co-occurrence` reports how often a second label gets added, which is the
cheapest way to find out whether nested screenshots are common enough to matter.

    python import_corpus.py              # dry run (the default)
    python import_corpus.py --apply
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from collections import Counter

from manifest import (CLASSES, IMPORTED, NEEDS_DETAIL, NEGATIVE, connect,
                      image_path, default_db,
                      default_images)

HF_DATASET = "howdyaendra/xblock-social-screenshots"
SOURCE = "imagefolder-v1"
HUMAN = "human"

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


RESOLUTIONS_FILE = Path(__file__).parent / "conflict_resolutions.json"


def load_resolutions(path: Path = RESOLUTIONS_FILE) -> dict[str, list[str]]:
    """Human decisions on CIDs whose folders contradict each other.

    Recorded in the repo rather than left to the review UI so the judgement is
    versioned, reviewable in a diff, and survives rebuilding the manifest from
    scratch. Without it these images would go back into the review queue every
    time the database is recreated.
    """
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f).get("resolutions", {})


def resolve_labels(cid: str, labels: set[str],
                   resolutions: dict[str, list[str]] | None = None
                   ) -> tuple[set[str], str, str, str]:
    """Decide the final labels, source and review state for one imported CID.

    Returns (labels, source, state, kind) where kind is one of
    'single' | 'multi' | 'resolved' | 'conflict'.

    A CID in two folders carries both labels -- except when one is `negative`,
    which means "not a screenshot of a post" and so cannot hold alongside a
    platform label. That is a filing error, not a co-occurrence, and nothing in
    the data says which folder was wrong. If a human has recorded a decision,
    use it; otherwise import with no label so the image stays out of training,
    and queue it for review.
    """
    if NEGATIVE in labels and len(labels) > 1:
        decided = (resolutions or {}).get(cid)
        if decided:
            # source=human because a person really did decide this. state stays
            # IMPORTED, not DONE: the decision settled which of two contradictory
            # folders was right, and said nothing about whether a second platform
            # also appears in the image. Marking it DONE would assert "this label
            # and no other" -- a claim nobody made -- and would permanently
            # remove it from the review queue.
            return set(decided), HUMAN, IMPORTED, "resolved"
        return set(), SOURCE, NEEDS_DETAIL, "conflict"
    return set(labels), SOURCE, IMPORTED, ("multi" if len(labels) > 1 else "single")


def main() -> None:
    here = Path(__file__).parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=default_db())
    p.add_argument("--images", default=default_images())
    p.add_argument("--dataset", default=HF_DATASET)
    p.add_argument("--dry-run", action="store_true",
                   help="report without writing (this is the default)")
    p.add_argument("--apply", action="store_true", help="write (default is a dry run)")
    args = p.parse_args()

    if args.dry_run and args.apply:
        sys.exit("--dry-run and --apply are contradictory; a dry run is the default")

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

    # Pass 1: collect the label SET per CID. A CID in two folders carries both
    # labels, so deduplicating on first sight would silently drop the second.
    labels_by_cid: dict[str, set[str]] = {}
    bytes_by_cid: dict[str, bytes] = {}
    skipped_dropped = skipped_nocid = 0

    for row in ds:
        label_name = mapping.get(row["label"])
        if label_name is None:
            skipped_dropped += 1
            continue

        blob = row["image"]
        raw, path = blob["bytes"], blob.get("path") or ""
        cid = cid_from_filename(path)
        if cid is None:
            if not raw:
                skipped_nocid += 1
                continue
            cid = "sha256" + hashlib.sha256(raw).hexdigest()[:40]

        labels_by_cid.setdefault(cid, set()).add(label_name)
        if raw and cid not in bytes_by_cid:
            bytes_by_cid[cid] = raw

    # Pass 2: resolve and write.
    counts: dict[str, int] = {}
    imported = skipped_existing = 0
    conflicts: list[tuple[str, list[str]]] = []
    resolved: list[tuple[str, list[str], list[str]]] = []
    multi: list[tuple[str, list[str]]] = []
    resolutions = load_resolutions()

    for cid, labels in labels_by_cid.items():
        if cid in existing:
            skipped_existing += 1
            continue

        original_labels = sorted(labels)
        labels, source, state, kind = resolve_labels(cid, labels, resolutions)
        if kind == "conflict":
            conflicts.append((cid, original_labels))
        elif kind == "resolved":
            resolved.append((cid, original_labels, sorted(labels)))
        elif kind == "multi":
            multi.append((cid, original_labels))

        if args.apply:
            dest = image_path(images_root, cid)
            dest.parent.mkdir(parents=True, exist_ok=True)
            raw = bytes_by_cid.get(cid)
            if raw:
                dest.write_bytes(raw)
            conn.execute(
                "INSERT OR IGNORE INTO images (cid, did, rkey, post_uri, path, bytes,"
                " harvested_at, bucket) VALUES (?,?,?,?,?,?,?,?)",
                (cid, "", "", "", str(dest.relative_to(images_root)),
                 len(raw or b""), now, IMPORTED))
            for label_name in labels:
                conn.execute(
                    "INSERT OR REPLACE INTO labels (cid,label,source,created_at)"
                    " VALUES (?,?,?,?)", (cid, label_name, source, now))
            conn.execute(
                "INSERT INTO review_state (cid,state,updated_at) VALUES (?,?,?) "
                "ON CONFLICT(cid) DO UPDATE SET state=excluded.state",
                (cid, state, now))

        existing.add(cid)
        for label_name in labels:
            counts[label_name] = counts.get(label_name, 0) + 1
        imported += 1

    if args.apply:
        conn.commit()
    conn.close()

    verb = "imported" if args.apply else "would import"
    print(f"\n{verb} {imported:,} images")
    for name in sorted(counts, key=lambda n: -counts[n]):
        print(f"  {name:<14}{counts[name]:>6,}")
    print(f"\nskipped: {skipped_dropped:,} dropped-class files, "
          f"{skipped_existing:,} already present, {skipped_nocid:,} unidentifiable")

    print(f"\nmulti-label (filed in two folders): {len(multi)}")
    for cid, ls in multi:
        print(f"  {' + '.join(ls)}")
    if resolved:
        print(f"\nresolved from conflict_resolutions.json: {len(resolved)}")
        for pair, n in sorted(Counter(
                f"{' + '.join(o)}  ->  {' + '.join(f)}" for _, o, f in resolved).items(),
                key=lambda kv: -kv[1]):
            print(f"  {pair:<40}{n:>4}")

    print(f"\nunresolved contradictions (platform + negative): {len(conflicts)}")
    for pair, n in sorted(Counter(" + ".join(ls) for _, ls in conflicts).items(),
                          key=lambda kv: -kv[1]):
        print(f"  {pair:<28}{n:>4}")
    if conflicts:
        print("  imported with NO label and queued for review -- negative means")
        print("  'not a screenshot', so it cannot hold alongside a platform label.")
    if not args.apply:
        print("\ndry run -- re-run with --apply to write")
    else:
        print(f"\nlabels written with source='{SOURCE}', review_state='{IMPORTED}'.")
        print("These are 'at least' labels. Re-review with the `imported` bucket "
              "filter, then `python dataset.py --co-occurrence` to see how often a "
              "second label gets added.")


if __name__ == "__main__":
    main()
