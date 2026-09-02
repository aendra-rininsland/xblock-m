#!/usr/bin/env python
"""Dead man's switch for the labeller.

The documented way to tell whether the labeller is alive is to open a websocket
by hand and count labels. That only runs when someone remembers, which means the
failure it detects is always found late. This is the same check, on a timer,
reporting to something that shouts when it stops hearing from it.

It watches the PUBLIC label stream, so it needs no access to any of the
machines. That is the point: it tests the thing users actually depend on -- that
labels are reaching the network -- rather than that a process is running. A
worker that is up but silently emitting nothing looks healthy to a process
monitor and dead to this.

This does NOT replace processor/watchdog.py, which already pings healthchecks.io
via HEALTHCHECKS_URL in processor/.env. That one watches the infrastructure --
worker process alive, jobs being consumed, firehose heartbeat fresh -- and all
of those can be green while zero labels reach the network. The two are
complementary, so give this its own check UUID: folding them into one check
would let an infrastructure ping mask a labelling outage.

Reads LABEL_HEALTHCHECKS_URL from processor/.env, following the same convention.

Two conditions, because there are two failure modes worth separating:

  rate     labels/hour across the window. Catches the whole pipeline stopping:
           the worker, Redis, the firehose consumer, or the model.
  classes  distinct platform classes seen. Catches partial collapse -- the
           retrain took this from 3 classes to 8, and a regression back toward
           twitter-only would keep the rate up while the model quietly got worse.

Set the thresholds against the TROUGH, not the average. Measured over 2 Sep
2026, output is strongly diurnal -- roughly 1,150 labels/hour at the US evening
peak against 125/hour in the small hours, and label yield per thousand images
swings 73 to 16 across the same day. That is content mix, not the model getting
worse. A threshold set from daytime numbers pages every night.

The window has to be long for the same reason. Labels arrive in bursts, so a
90-second sample of a healthy 286/hour stream can legitimately return one label
-- which reads exactly like an outage. Five minutes is the floor; ten is better.

Exit codes: 0 healthy, 1 below threshold, 2 could not connect.

    python label-watchdog.py --window 600 --min-labels 5 --min-classes 1 \
        --ping-url https://hc-ping.com/<uuid>
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

STREAM = "wss://xblock.aendra.dev/xrpc/com.atproto.label.subscribeLabels"

# Matched explicitly rather than with a loose [a-z]+-screenshot pattern: the
# frames are CBOR, so a loose regex picks up preceding bytes as part of the
# class name and every count lands under a slightly different key.
CLASSES = ["twitter", "bluesky", "threads", "discord", "facebook", "instagram",
           "reddit", "altright", "tumblr", "fediverse", "ngl"]
PATTERN = re.compile(("(" + "|".join(CLASSES) + ")-screenshot").encode())


async def sample(url: str, window: int) -> tuple[collections.Counter, bool]:
    """Count labels by class for `window` seconds. Returns (counts, connected)."""
    try:
        import websockets
    except ImportError:
        sys.exit("pip install websockets")

    counts: collections.Counter = collections.Counter()
    started = time.monotonic()
    try:
        async with websockets.connect(url, open_timeout=30) as ws:
            while True:
                remaining = window - (time.monotonic() - started)
                if remaining <= 0:
                    break
                try:
                    frame = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                for match in PATTERN.findall(frame):
                    counts[match.decode()] += 1
    except Exception as exc:
        print(f"connect failed: {exc}", file=sys.stderr)
        return counts, False
    return counts, True


def ping(url: str, ok: bool, body: str) -> None:
    """Report to a dead man's switch endpoint.

    Healthchecks.io and Uptime Kuma both take a plain GET, with /fail appended
    for an explicit failure. A failure ping is better than staying silent: it
    turns a threshold breach into an alert now rather than at the grace period.
    """
    target = url if ok else url.rstrip("/") + "/fail"
    try:
        req = urllib.request.Request(target, data=body.encode()[:10000],
                                     headers={"Content-Type": "text/plain"})
        urllib.request.urlopen(req, timeout=15).read()
    except Exception as exc:
        print(f"ping failed ({target}): {exc}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=STREAM)
    p.add_argument("--window", type=int, default=600,
                   help="seconds to listen (default 600; below 300 is too "
                        "bursty to draw a conclusion from)")
    p.add_argument("--min-labels", type=int, default=5,
                   help="fail below this many labels in the window. Default 5 "
                        "sits well under the ~21 expected at the quietest hour "
                        "measured (125/hr over 600s)")
    p.add_argument("--min-classes", type=int, default=1,
                   help="fail below this many distinct classes. 1 detects a "
                        "total stop; raise to 2-3 only with a longer window, "
                        "since a quiet hour can legitimately be twitter-only")
    p.add_argument("--ping-url", default=None,
                   help="Healthchecks.io or Uptime Kuma push URL. Defaults to "
                        "LABEL_HEALTHCHECKS_URL from the environment or "
                        "processor/.env. Must be a DIFFERENT check from the "
                        "HEALTHCHECKS_URL that processor/watchdog.py uses.")
    args = p.parse_args()

    if not args.ping_url:
        args.ping_url = os.getenv("LABEL_HEALTHCHECKS_URL") or None
    if not args.ping_url:
        # Same file processor/watchdog.py loads, so one place holds the config.
        env = Path(__file__).resolve().parent.parent / "processor" / ".env"
        if env.is_file():
            for line in env.read_text(errors="replace").splitlines():
                key, _, value = line.partition("=")
                if key.strip() == "LABEL_HEALTHCHECKS_URL":
                    args.ping_url = value.strip().strip("'\"") or None
                    break

    counts, connected = asyncio.run(sample(args.url, args.window))
    total = sum(counts.values())
    classes = len([c for c, n in counts.items() if n])
    rate = total / args.window * 3600 if args.window else 0

    lines = [
        f"window={args.window}s labels={total} rate={rate:.0f}/hr classes={classes}",
        "  " + ("  ".join(f"{k}={v}" for k, v in counts.most_common()) or "(none)"),
    ]

    if not connected:
        status, code = "UNREACHABLE", 2
    elif total < args.min_labels or classes < args.min_classes:
        status, code = "BELOW THRESHOLD", 1
    else:
        status, code = "OK", 0
    lines.insert(0, f"{status}: labeller {args.url}")

    report = "\n".join(lines)
    print(report)
    if args.ping_url:
        ping(args.ping_url, code == 0, report)
    sys.exit(code)


if __name__ == "__main__":
    main()
