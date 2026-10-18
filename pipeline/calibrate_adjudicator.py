#!/usr/bin/env python
"""
Measure the adjudicator against human-labelled ground truth, before it is
allowed anywhere near the moderation queue.

Why this runs first
-------------------
Every confidence threshold in the triage design is a guess until this has run.
Worse, without it there is no way to tell a working adjudicator from a broken
one: a broken one agrees with everything, closes the queue, and reports no
errors. This produces the per-class accuracy the thresholds are set from, and
it is also the go/no-go -- if Sonnet cannot beat the classifier on classes the
classifier is bad at, the premise fails and nothing else should be built.

Ground truth
------------
Labelled images from the manifest, ignoring the train/val/test split.

The split exists to hold data back from the CLASSIFIER. Sonnet never saw any of
it, so for measuring Sonnet the split is irrelevant -- and using only `test`
would be actively harmful: test holds 633 images with roughly 23 platform
positives across all eleven classes (six twitter, four discord, one reddit),
which cannot support a per-class number.

The two label sources are measured SEPARATELY and never merged, because they
are not equally trustworthy:

  human           4,829 negatives but only ~209 platform positives, and the
                  classes that matter most are the thinnest -- reddit 5,
                  tumblr 5, ngl 5, fediverse 0. Verified, and far too few.
  imagefolder-v1  the bulk: twitter 718, bluesky 118, threads 70, reddit 39,
                  fediverse 7. Imported from a folder layout and never reviewed
                  here -- every one is still in state `imported`.

Reported apart, the difference is informative. High agreement with human labels
and low agreement with imagefolder ones means the imagefolder labels are noisy,
not that the adjudicator is bad. Averaged together, that signal disappears and
a per-class threshold gets set off a number that means nothing.

Sampling is stratified and capped per class: the corpus runs 4,992 negatives
against 7 fediverse, so an unstratified sample measures almost nothing but
negatives.

An image is scored as agreeing when the predicted platform is among the labels
that image carries. Only five images carry more than one label, but discarding
them would have cost most of the reddit data.

Nothing here writes to Ozone, to the manifest, or anywhere else. It reads
images and prints a table.

    python calibrate_adjudicator.py --per-class 30 --dry-run   # cost estimate
    python calibrate_adjudicator.py --per-class 30
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from adjudicate import MODEL, PLATFORMS, adjudicate  # noqa: E402
from manifest import connect, default_db, default_images  # noqa: E402

# Sonnet 5.5, $/MTok. Used only for the estimate; verify against current
# pricing before relying on a figure.
PRICE_IN, PRICE_OUT = 2.00, 10.00
EST_IN_TOK, EST_OUT_TOK = 2000, 250


def ground_truth(db: sqlite3.Connection, source: str, per_class: int,
                 seed: int) -> list[tuple]:
    """Stratified sample: (cid, path, {labels}, primary_label, source).

    Stratified on the primary label -- the single non-negative label where there
    is one, else the only label -- so the per-class caps mean what they say.
    """
    rows = db.execute("""
        SELECT l.cid, i.path, l.label, l.source
          FROM labels l
          JOIN images i ON i.cid = l.cid
         WHERE l.source = ?
    """, (source,)).fetchall()

    per_image: dict[str, dict] = {}
    for cid, path, label, src in rows:
        entry = per_image.setdefault(cid, {"path": path, "labels": set(), "source": src})
        entry["labels"].add(label)

    by_class: dict[str, list] = collections.defaultdict(list)
    for cid, e in per_image.items():
        non_negative = sorted(e["labels"] - {"negative"})
        primary = non_negative[0] if non_negative else "negative"
        by_class[primary].append((cid, e["path"], e["labels"], primary, e["source"]))

    rng = random.Random(seed)
    sample = []
    for label in sorted(by_class):
        pool = sorted(by_class[label])          # deterministic before shuffling
        rng.shuffle(pool)
        sample.extend(pool[:per_class])
    rng.shuffle(sample)
    return sample


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=default_db())
    p.add_argument("--images", default=default_images())
    p.add_argument("--source", default="human",
                   choices=["human", "imagefolder-v1"],
                   help="which label source to measure against. Run both and "
                        "compare: they are not equally trustworthy and merging "
                        "them hides which one the disagreement is with")
    p.add_argument("--per-class", type=int, default=30,
                   help="cap per class; rare classes contribute all they have")
    p.add_argument("--seed", type=int, default=20261018,
                   help="fixed so re-runs are comparable")
    p.add_argument("--dry-run", action="store_true",
                   help="show the sample and the cost estimate, call nothing")
    p.add_argument("--out", default="calibration.json")
    args = p.parse_args()

    db = connect(args.db)
    sample = ground_truth(db, args.source, args.per_class, args.seed)
    images_root = Path(args.images)

    counts = collections.Counter(primary for _, _, _, primary, _ in sample)
    print(f"sample: {len(sample)} images across {len(counts)} classes "
          f"from source={args.source} (cap {args.per_class}/class)")
    for label, n in counts.most_common():
        print(f"  {label:<14} {n}")

    est = len(sample) * (EST_IN_TOK / 1e6 * PRICE_IN + EST_OUT_TOK / 1e6 * PRICE_OUT)
    print(f"\nestimated cost: ~${est:.2f} at {MODEL} list rates "
          f"(~{EST_IN_TOK} in / {EST_OUT_TOK} out per image)")

    if args.dry_run:
        print("\ndry run -- nothing called")
        return

    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
            or Path.home().joinpath(".config/anthropic").is_dir()):
        sys.exit("no Anthropic credentials found: set ANTHROPIC_API_KEY or run `ant auth login`")

    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")

    client = anthropic.Anthropic()

    # confusion[true][predicted]; agreement by class; confidence buckets
    confusion: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    conf_right: dict[str, list[float]] = collections.defaultdict(list)
    conf_wrong: dict[str, list[float]] = collections.defaultdict(list)
    failures: list[dict] = []
    records: list[dict] = []
    started = time.time()

    for n, (cid, path, truth_set, truth, _src) in enumerate(sample, 1):
        image = images_root / path
        try:
            j = adjudicate(client, image)
        except FileNotFoundError:
            failures.append({"cid": cid, "error": "image missing"})
            continue
        except Exception as exc:                      # noqa: BLE001
            failures.append({"cid": cid, "error": f"{type(exc).__name__}: {exc}"})
            continue

        # The corpus calls a non-platform image 'negative'; the adjudicator
        # distinguishes 'uncategorised' from 'not-a-screenshot'. Both mean "no
        # platform label", so both count as agreeing with 'negative'.
        # Agreement = the prediction is among the labels this image carries.
        # 'negative' in the corpus means "no platform"; the adjudicator splits
        # that into 'uncategorised' and 'not-a-screenshot', so both count.
        predicted = j.platform
        agreed = predicted in truth_set or (
            "negative" in truth_set and predicted in ("uncategorised", "not-a-screenshot"))

        confusion[truth][predicted] += 1
        (conf_right if agreed else conf_wrong)[truth].append(j.confidence)
        records.append({"cid": cid, "truth": truth,
                        "truth_labels": sorted(truth_set), "predicted": predicted,
                        "confidence": j.confidence, "agreed": agreed,
                        "evidence": j.evidence})

        if n % 20 == 0:
            ok = sum(1 for r in records if r["agreed"])
            print(f"  {n}/{len(sample)}  agreement so far {ok}/{len(records)} "
                  f"({100*ok/len(records):.0f}%)", end="\r", flush=True)

    print()
    elapsed = time.time() - started

    print(f"\n=== agreement with human labels ({len(records)} scored, "
          f"{len(failures)} failed, {elapsed:.0f}s) ===")
    print(f"{'class':<14}{'n':>5}{'agreed':>8}{'rate':>8}   "
          f"{'conf when right':>16}{'conf when wrong':>17}")
    for truth in sorted(confusion):
        total = sum(confusion[truth].values())
        right = len(conf_right[truth])
        cr = sum(conf_right[truth]) / right if right else float("nan")
        wrong = conf_wrong[truth]
        cw = sum(wrong) / len(wrong) if wrong else float("nan")
        print(f"{truth:<14}{total:>5}{right:>8}{100*right/total:>7.0f}%   "
              f"{cr:>16.2f}{cw:>17.2f}")

    # The number that decides whether confidence is usable as a gate at all.
    # If the model is as confident when wrong as when right, no threshold
    # separates them and the whole design needs rethinking.
    all_right = [c for v in conf_right.values() for c in v]
    all_wrong = [c for v in conf_wrong.values() for c in v]
    if all_right and all_wrong:
        print(f"\nmean confidence when right: {sum(all_right)/len(all_right):.3f}")
        print(f"mean confidence when wrong: {sum(all_wrong)/len(all_wrong):.3f}")
        print("  (these need to be clearly separated for a confidence gate to work;"
              "\n   if they are not, the gate is theatre and the design needs rethinking)")

    print("\n=== where it disagreed (true -> predicted) ===")
    for truth in sorted(confusion):
        for pred, n in confusion[truth].most_common():
            if pred != truth and not (truth == "negative"
                                      and pred in ("uncategorised", "not-a-screenshot")):
                print(f"  {truth:<14} -> {pred:<18} {n}")

    if failures:
        print(f"\n=== {len(failures)} failures ===")
        for f in failures[:10]:
            print(f"  {f['cid'][:20]}  {f['error']}")

    Path(args.out).write_text(json.dumps(
        {"model": MODEL, "source": args.source, "sample": len(sample),
         "scored": len(records), "seed": args.seed,
         "records": records, "failures": failures}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
