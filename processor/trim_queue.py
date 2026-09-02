#!/usr/bin/env python
"""
Drop waiting jobs that are too old to be worth processing.

Why a separate tool
-------------------
worker.py skips stale jobs as it pops them, which stops wasted inference. But
the queue is LIFO: BullMQ pushes new jobs to the tail and the worker takes from
the tail, so the OLDEST jobs sit at the head and are never reached until the
queue fully drains. They are not costing CPU -- they are costing gigabytes of
Redis, indefinitely. Only something that reaches into the head reclaims that.

This is an age bound, deliberately not a depth cap. A depth cap would discard the
newest work during a burst, which is exactly the diurnal buffering the LIFO
design exists to provide. An age bound only discards work whose value has already
expired: a Bluesky post has a half-life of hours, so a label applied days late
reaches almost nobody.

Safety
------
Dry run by default; --apply is required to delete anything. Before deleting, the
head/tail ordering assumption is verified against the live queue -- if the head
is NOT older than the tail, the LIFO assumption is wrong for this queue and
trimming the head would destroy the newest jobs, so the tool refuses to run.

    python trim_queue.py                    # report only
    python trim_queue.py --apply
    python trim_queue.py --apply --max-age-hours 24
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import redis
from dotenv import load_dotenv

from constants import MAX_JOB_AGE_HOURS

# worker.py does this and this file did not, so REDIS_CONNECTION_STRING was only
# ever visible if it happened to be exported -- and under supervisord it is not.
# The result was a program that exited immediately with "set
# REDIS_CONNECTION_STRING", every time, which is why the age bound has never
# actually been enforced. The connection string carries a password now, so it
# belongs in processor/.env rather than in the tracked supervisord.conf.
load_dotenv()

BATCH = 500

# Below this, head and tail are treated as the same age rather than as an
# ordering violation. See check_ordering.
ORDERING_TOLERANCE_MS = 5 * 60 * 1000


def job_key(prefix: str, queue: str, job_id: str) -> str:
    return f"{prefix}:{queue}:{job_id}"


def read_timestamps(r: redis.Redis, prefix: str, queue: str,
                    job_ids: list[str]) -> list[float | None]:
    pipe = r.pipeline()
    for jid in job_ids:
        pipe.hget(job_key(prefix, queue, jid), "timestamp")
    out = []
    for raw in pipe.execute():
        try:
            out.append(float(raw))
        except (TypeError, ValueError):
            out.append(None)   # hash already gone, or malformed
    return out


def check_ordering(r: redis.Redis, prefix: str, queue: str,
                   wait_key: str) -> tuple[bool, bool, str]:
    """Confirm the head really is older than the tail before deleting from it.

    If this queue were FIFO, the head would hold the NEWEST jobs and trimming it
    would throw away everything just enqueued. Refuse rather than guess.

    Returns (ok, fatal, message). `fatal` separates "the ordering assumption is
    violated, stop entirely" from "there is nothing here to verify against",
    which is benign and simply means there is nothing worth trimming.
    """
    depth = r.llen(wait_key)
    if depth < 100:
        return False, False, f"queue too short to verify ordering ({depth} jobs); nothing to do"

    head = r.lrange(wait_key, 0, 19)
    tail = r.lrange(wait_key, -20, -1)
    head_ts = [t for t in read_timestamps(r, prefix, queue, head) if t]
    tail_ts = [t for t in read_timestamps(r, prefix, queue, tail) if t]
    if not head_ts or not tail_ts:
        return False, False, "could not read timestamps from head/tail; skipping"

    head_avg, tail_avg = sum(head_ts) / len(head_ts), sum(tail_ts) / len(tail_ts)

    # Near-equal timestamps are normal, not a violation: a small queue filled in
    # one burst has a head and tail seconds apart, and floating-point noise can
    # put them in either order. Only a CLEAR inversion means the LIFO assumption
    # is wrong. Treating equality as fatal killed the daemon the moment the queue
    # was trimmed down to a batch of same-second jobs.
    if abs(head_avg - tail_avg) < ORDERING_TOLERANCE_MS:
        return False, False, (
            f"head and tail are within {ORDERING_TOLERANCE_MS / 60000:g} min of each "
            "other; ordering indeterminate, skipping this pass")

    if head_avg > tail_avg:
        return False, True, (
            "head is NOT older than tail "
            f"(head {time.ctime(head_avg / 1000)}, tail {time.ctime(tail_avg / 1000)}). "
            "This queue does not look LIFO -- trimming the head would delete the "
            "newest jobs. Refusing."
        )
    gap_h = (tail_avg - head_avg) / 3_600_000
    return True, False, f"head is {gap_h:.1f}h older than tail, as expected for LIFO"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=os.getenv("REDIS_CONNECTION_STRING"))
    p.add_argument("--queue", default="xblock")
    p.add_argument("--prefix", default="bull")
    p.add_argument("--max-age-hours", type=float, default=MAX_JOB_AGE_HOURS)
    p.add_argument("--dry-run", action="store_true",
                   help="report without writing (this is the default)")
    p.add_argument("--apply", action="store_true", help="actually delete (default is a dry run)")
    p.add_argument("--limit", type=int, default=0, help="stop after N deletions (0 = no limit)")
    p.add_argument("--interval", type=int, default=0,
                   help="run forever, trimming every N seconds (for supervisord)")
    args = p.parse_args()

    if args.dry_run and args.apply:
        sys.exit("--dry-run and --apply are contradictory; a dry run is the default")

    # Line-buffer stdout. Without this the per-pass output sits in a 4KB buffer
    # and supervisord's trim.log stays empty for hours, which makes a daemon that
    # runs once an hour look dead.
    sys.stdout.reconfigure(line_buffering=True)

    if not args.url:
        sys.exit("set REDIS_CONNECTION_STRING or pass --url")

    r = redis.from_url(args.url, decode_responses=True, socket_timeout=15)

    if not args.interval:
        raise SystemExit(0 if trim_once(r, args) is not None else 1)

    # Daemon mode, matching watchdog.py's shape so supervisord can own it.
    print(f"trim daemon: every {args.interval}s, dropping jobs older than "
          f"{args.max_age_hours:g}h", flush=True)
    while True:
        try:
            if trim_once(r, args) is None:
                raise SystemExit(1)   # ordering violation: stop, do not keep trying
        except redis.RedisError as e:
            print(f"redis error: {e}", flush=True)
        time.sleep(args.interval)


def trim_once(r: redis.Redis, args) -> int | None:
    """One trim pass. Returns jobs removed, or None if it refused on safety."""
    wait_key = f"{args.prefix}:{args.queue}:wait"

    depth = r.llen(wait_key)
    cutoff_ms = (time.time() - args.max_age_hours * 3600) * 1000
    print(f"queue     {wait_key}")
    print(f"depth     {depth:,}")
    print(f"cutoff    older than {args.max_age_hours:g}h ({time.ctime(cutoff_ms / 1000)})")
    if not depth:
        return 0

    # Ordering guard first. Deleting a contiguous stale prefix is safe whatever
    # the order, so this is defence in depth -- but pointing the tool at a queue
    # that is not LIFO should say so out loud rather than silently doing nothing.
    ok, fatal, why = check_ordering(r, args.prefix, args.queue, wait_key)
    print(f"ordering  {why}")
    if not ok:
        return None if fatal else 0

    # Nothing at the head is stale -> nothing to delete.
    oldest = read_timestamps(r, args.prefix, args.queue, r.lrange(wait_key, 0, 0))
    if oldest and oldest[0] is not None and oldest[0] >= cutoff_ms:
        print("head      oldest job is within the age bound; nothing to do")
        return 0
    print(f"mode      {'APPLY -- deleting' if args.apply else 'dry run'}\n")

    removed = 0
    started = time.time()
    while True:
        if args.limit and removed >= args.limit:
            print(f"\nstopping at --limit {args.limit}")
            break

        head = r.lrange(wait_key, 0, BATCH - 1)
        if not head:
            break

        stale = 0
        for ts in read_timestamps(r, args.prefix, args.queue, head):
            # A missing timestamp means the hash is already gone; treat the entry
            # as removable rather than letting one orphan block the whole scan.
            if ts is None or ts < cutoff_ms:
                stale += 1
            else:
                break   # head is oldest-first, so the first fresh job ends it

        if stale == 0:
            break
        if args.limit:
            stale = min(stale, args.limit - removed)

        if args.apply:
            ids = head[:stale]
            pipe = r.pipeline()
            pipe.ltrim(wait_key, stale, -1)
            for jid in ids:
                pipe.delete(job_key(args.prefix, args.queue, jid))
            pipe.execute()

        removed += stale
        if removed % 10_000 < BATCH:
            rate = removed / max(time.time() - started, 1)
            print(f"  {'removed' if args.apply else 'would remove'} {removed:,}"
                  f" ({rate:,.0f}/s)")

        if not args.apply:
            # Nothing was deleted, so the head has not moved and the loop would
            # not terminate. One batch is enough to report on.
            print(f"\ndry run: {removed:,} of the oldest {BATCH:,} jobs are stale.")
            print("Re-run with --apply to delete; the scan continues past this batch.")
            return removed

    if args.apply:
        after = r.llen(wait_key)
        print(f"\nremoved   {removed:,}")
        print(f"depth     {depth:,} -> {after:,}")
        try:
            used = r.info("memory").get("used_memory_human")
            print(f"memory    {used} (Redis may not return freed pages to the OS "
                  f"immediately; the drop shows in used_memory, not RSS)")
        except redis.RedisError:
            pass
    return removed


if __name__ == "__main__":
    main()
