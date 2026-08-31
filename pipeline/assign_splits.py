#!/usr/bin/env python
"""
Assign train/val/test splits in the manifest, once and durably.

Why this exists rather than splitting at training time
------------------------------------------------------
The previous notebook computed its split inside train(), and its test set was the
entire corpus -- so the reported TEST AUROC was ~90% training accuracy and came
out above validation. A split computed at training time is a split that can drift
between runs, which makes two models incomparable even when nothing else changed.

So membership lives in the manifest, in its own table, and this tool is
**idempotent: an image that already has a split is never reassigned.** That is
what keeps a frozen evaluation set frozen as the corpus grows -- new images fill
the remaining need, existing assignments are untouched.

Stratification
--------------
Labels are sets, so ordinary stratification does not apply. This uses the usual
greedy approximation of iterative stratification: process images rarest-label
first, and send each to whichever split most needs that label. Rare classes are
placed while there is still freedom to place them, which is what stops `ngl` (6
images) from landing entirely in train.

It is an approximation, not the full Sechidis algorithm. With classes this small
the per-class report at the end matters more than the method -- read it.

    python assign_splits.py                      # dry run
    python assign_splits.py --apply
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dataset import read_rows
from manifest import CLASSES, connect


def plan_splits(rows: list[dict], targets: dict[str, float], seed: int) -> dict[str, str]:
    """Greedy iterative stratification. Returns {cid: split}."""
    rng = random.Random(seed)
    rows = sorted(rows, key=lambda r: r["cid"])
    rng.shuffle(rows)

    freq = Counter(l for r in rows for l in r["labels"])
    # Rarest label first: a class with six examples has to be placed while there
    # is still room to place it.
    rows.sort(key=lambda r: min((freq[l] for l in r["labels"]), default=10**9))

    need: dict[str, dict[str, float]] = {
        s: {c: freq[c] * frac for c in CLASSES} for s, frac in targets.items()
    }
    total_need = {s: len(rows) * frac for s, frac in targets.items()}
    out: dict[str, str] = {}

    # Floor pass. At 80/10/10 a six-image class rounds to 0.6 expected in each of
    # val and test, and greedy assignment leaves one of them empty -- which makes
    # the class invisible to evaluation rather than merely noisy. Reserve one
    # image per split for every class that can spare it, before the greedy pass
    # takes them. Costs one training example each; buys a class you can measure
    # at all.
    MIN_PER_SPLIT_IF_AT_LEAST = 3
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        for l in r["labels"]:
            by_class[l].append(r)

    for cls in sorted(by_class, key=lambda c: freq[c]):
        if freq[cls] < MIN_PER_SPLIT_IF_AT_LEAST:
            continue
        for split in targets:
            if split == max(targets, key=lambda s: targets[s]):
                continue   # the majority split never needs a floor
            if any(out.get(r["cid"]) == split for r in by_class[cls]):
                continue
            # Prefer an image labelled ONLY with this class, so reserving it
            # does not drag some other class's balance around with it.
            pick = next((r for r in by_class[cls]
                         if r["cid"] not in out and len(r["labels"]) == 1), None)
            pick = pick or next((r for r in by_class[cls] if r["cid"] not in out), None)
            if pick is None:
                continue
            out[pick["cid"]] = split
            for l in pick["labels"]:
                need[split][l] -= 1
            total_need[split] -= 1

    for r in rows:
        if r["cid"] in out:
            continue
        rarest = min(r["labels"], key=lambda l: freq[l]) if r["labels"] else None
        if rarest is None:
            best = max(total_need, key=lambda s: total_need[s])
        else:
            # Most outstanding need for the rarest label, ties broken by overall
            # remaining capacity so the split sizes stay honest.
            best = max(need, key=lambda s: (need[s][rarest], total_need[s]))
        out[r["cid"]] = best
        for l in r["labels"]:
            need[best][l] -= 1
        total_need[best] -= 1
    return out


def main() -> None:
    here = Path(__file__).parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=str(here / "data" / "pipeline.db"))
    p.add_argument("--images", default=str(here / "data" / "images"))
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--apply", action="store_true", help="write (default is a dry run)")
    args = p.parse_args()

    targets = {"train": args.train, "val": args.val, "test": args.test}
    total = sum(targets.values())
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"--train/--val/--test must sum to 1.0, got {total}")

    conn = connect(args.db)
    assigned = {r[0]: r[1] for r in conn.execute("SELECT cid, split FROM splits")}
    rows = read_rows(args.db, Path(args.images))
    if not rows:
        raise SystemExit(f"{args.db} has no labelled images to split")

    pending = [r for r in rows if r["cid"] not in assigned]
    print(f"labelled images   {len(rows):,}")
    print(f"already assigned  {len(assigned):,}   (never reassigned)")
    print(f"to assign         {len(pending):,}\n")
    if not pending:
        print("nothing to do")
        return

    plan = plan_splits(pending, targets, args.seed)

    now = datetime.now(timezone.utc).isoformat()
    if args.apply:
        for cid, split in plan.items():
            conn.execute(
                "INSERT INTO splits (cid, split, assigned_at) VALUES (?,?,?) "
                "ON CONFLICT(cid) DO NOTHING", (cid, split, now))
        conn.commit()

    # Report the resulting per-class distribution across ALL assignments, not
    # just this run's -- that is what actually determines whether a class is
    # measurable in the test split.
    final = dict(assigned)
    final.update(plan)
    by_cid = {r["cid"]: r["labels"] for r in rows}
    per = defaultdict(Counter)
    for cid, split in final.items():
        for l in by_cid.get(cid, []):
            per[l][split] += 1

    width = max(len(c) for c in CLASSES)
    print(f"{'class':<{width}}{'train':>8}{'val':>7}{'test':>7}   note")
    for c in CLASSES:
        counts = per[c]
        note = ""
        if counts["test"] == 0 and sum(counts.values()):
            note = "NOT MEASURABLE — nothing in test"
        elif 0 < counts["test"] < 8:
            note = "too few in test to mean much"
        print(f"{c:<{width}}{counts['train']:>8,}{counts['val']:>7,}"
              f"{counts['test']:>7,}   {note}")

    sizes = Counter(final.values())
    print(f"\ntotals: " + "  ".join(f"{s}={sizes[s]:,}" for s in ("train", "val", "test")))
    if not args.apply:
        print("\ndry run — re-run with --apply to write")
    conn.close()


if __name__ == "__main__":
    main()
