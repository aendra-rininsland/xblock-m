#!/usr/bin/env python
"""
Harvest images from the Bluesky firehose into a multi-label manifest.

Why this exists
---------------
The training corpus is an ImageFolder tree: one directory per class, so exactly
one label per image. That structurally cannot express a co-occurrence, which is
why the nested-screenshot case (a Bluesky post quoting a Twitter screenshot,
firing both) is unlearnable no matter how the model is trained.

It is also badly short of negatives -- roughly 0.17x the positive count, against
a production stream that is overwhelmingly not-a-screenshot. That mismatch is the
most likely driver of false positives in production.

This tool addresses both: it samples the live firehose into a SQLite manifest
where an image carries a *set* of labels, and where negatives are free.

Sampling
--------
Harvesting only what the model already scores low would fill the corpus with easy
negatives that teach nothing. Buckets:

  random     Unbiased sample of the production stream. The bulk of a harvest,
             and the only source of a true prior -- which is what a frozen eval
             set needs and a curated corpus cannot give you.
  fired      The model scores >= THRESHOLD on some class. These are candidate
             false positives: the negatives actually worth labelling.
  uncertain  Max non-negative score in the uncertainty band. Where the decision
             boundary is, so where new labels move it most.
  multi-cand Two or more classes scoring meaningfully. Nested screenshots -- the
             case the single-label corpus could never express -- which random
             sampling will essentially never turn up.

Bucketing needs --score (loads the model). Without it everything is `random`,
which is still the single most useful thing to collect right now.

Nothing here labels anything. Harvested images arrive unlabelled by design;
`export` produces a review queue.

Usage
-----
    python harvest.py harvest --target 4000 --rate 0.05
    python harvest.py harvest --target 2000 --score --bucket fired
    python harvest.py stats
    python harvest.py export --split unassigned --out review.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import random
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from manifest import connect, image_path, default_db, default_images

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("harvest")

JETSTREAM = os.getenv(
    "JETSTREAM_URL",
    "wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post",
)
CDN = "https://cdn.bsky.app/img/feed_thumbnail/plain/{did}/{cid}@jpeg"

UNCERTAIN_BAND = (0.35, 0.75)
# Second-highest non-negative score at or above this makes an image a candidate
# for carrying more than one label.
MULTI_CANDIDATE_FLOOR = 0.35
FETCH_CONCURRENCY = 16
MAX_IMAGE_BYTES = 8 * 1024 * 1024


# ── optional scoring ──────────────────────────────────────────────────────────

class Scorer:
    """Loads the serving model. Imported lazily so harvesting without --score
    needs neither torch nor a GPU."""

    def __init__(self, model_name: str):
        sys.path.insert(0, str(Path(__file__).parent.parent / "processor"))
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from timm import create_model
        # Shared with the worker so collection and serving cannot diverge.
        from preprocessing import build_transform

        self.torch = torch
        self.transform = build_transform()
        self.model_name = model_name

        repo = f"howdyaendra/{model_name}"
        weights = hf_hub_download(repo_id=repo, filename="model.safetensors")
        with open(hf_hub_download(repo_id=repo, filename="config.json")) as f:
            cfg = json.load(f)
        self.labels: list[str] = cfg["label_names"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = create_model("swin_s3_base_224", num_classes=len(self.labels), pretrained=False)
        model.load_state_dict(load_file(weights))
        model.to(self.device).eval()
        self.model = model
        log.info("Scorer ready: %s on %s (%d classes)", model_name, self.device, len(self.labels))

    def score(self, raw: bytes) -> dict[str, float] | None:
        from io import BytesIO

        from PIL import Image
        try:
            img = Image.open(BytesIO(raw)).convert("RGB")
        except Exception as e:
            log.warning("decode failed: %s", e)
            return None
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            probs = self.model(tensor).sigmoid().cpu().numpy()[0]
        return {name: float(p) for name, p in zip(self.labels, probs)}


def eligible_images(evt: dict) -> tuple[str, str, list[str]] | None:
    """Pull (did, rkey, [blob cids]) out of a Jetstream event, or None if the
    event is not one production would have processed.

    The filter mirrors lib/firehose.ts exactly -- image embeds, English-tagged,
    creates only -- so that a harvested sample reflects the distribution the
    model actually sees rather than the whole network.
    """
    commit = evt.get("commit") or {}
    if commit.get("operation") != "create":
        return None
    if commit.get("collection") != "app.bsky.feed.post":
        return None
    record = commit.get("record") or {}
    embed = record.get("embed") or {}
    if embed.get("$type") != "app.bsky.embed.images":
        return None
    if "en" not in (record.get("langs") or []):
        return None

    did, rkey = evt.get("did"), commit.get("rkey")
    if not did or not rkey:
        return None

    cids = []
    for img in embed.get("images") or []:
        cid = ((img.get("image") or {}).get("ref") or {}).get("$link")
        if cid:
            cids.append(cid)
    return (did, rkey, cids) if cids else None


def classify_bucket(scores: dict[str, float], threshold: float) -> str:
    """Which stratum this image belongs to, given the current model's opinion.

    multi-candidate is checked before fired: an image scoring twitter 0.95 and
    bluesky 0.4 is both a false-positive candidate and a co-occurrence candidate,
    and the second reading is the scarcer one. Nested screenshots are the case
    the single-label corpus could never teach, and random sampling will almost
    never surface them.

    Worth knowing this is partly circular -- a model trained only on single-label
    data is biased toward one confident class, so it under-reports exactly what
    is being mined. A genuinely nested image still presents visual evidence for
    both platforms, so it should surface them far better than chance, but the
    yield is worth measuring rather than assuming.
    """
    non_neg = sorted((v for k, v in scores.items() if k != "negative"), reverse=True)
    top = non_neg[0] if non_neg else 0.0
    second = non_neg[1] if len(non_neg) > 1 else 0.0
    if second >= MULTI_CANDIDATE_FLOOR:
        return "multi-candidate"
    if top >= threshold:
        return "fired"
    if UNCERTAIN_BAND[0] <= top <= UNCERTAIN_BAND[1]:
        return "uncertain"
    return "random"


# ── harvest ───────────────────────────────────────────────────────────────────

def looks_complete(data: bytes) -> bool:
    """Structural check that the bytes are a whole image, not a prefix of one.

    Truncation is the failure that matters here: a partial JPEG still decodes,
    it just renders the scanlines that arrived and fills the rest, so it lands in
    the corpus looking like a real image with most of it replaced by garbage.
    """
    if len(data) < 100:
        return False
    if data[:2] == b"\xff\xd8":                      # JPEG: needs its EOI marker
        return data.rstrip(b"\x00")[-2:] == b"\xff\xd9"
    if data[:8] == b"\x89PNG\r\n\x1a\n":              # PNG: needs IEND
        return b"IEND" in data[-16:]
    return True                                       # unknown type: don't guess


async def fetch(session: aiohttp.ClientSession, url: str) -> bytes | None:
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                return None
            if resp.content_length and resp.content_length > MAX_IMAGE_BYTES:
                log.warning("skipping oversized image (%s bytes)", resp.content_length)
                return None

            # NOT resp.content.read(n): StreamReader.read() returns UP TO n bytes
            # and returns as soon as anything is buffered, so it hands back the
            # first chunk and silently truncates the image. Read to EOF, capping
            # as we go.
            buf = bytearray()
            async for chunk in resp.content.iter_chunked(64 * 1024):
                buf.extend(chunk)
                if len(buf) > MAX_IMAGE_BYTES:
                    log.warning("oversized image, discarding: %s", url)
                    return None
            data = bytes(buf)

            # Content-Length is the definitive truncation check when the server
            # sends one; the structural check covers chunked responses that don't.
            if resp.content_length is not None and len(data) != resp.content_length:
                log.warning("short read %d/%d bytes: %s", len(data),
                            resp.content_length, url)
                return None
            if not looks_complete(data):
                log.warning("incomplete image discarded: %s", url)
                return None
            return data
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        log.debug("fetch failed %s: %s", url, e)
        return None


async def harvest(args: argparse.Namespace) -> None:
    conn = connect(args.db)
    images_root = Path(args.images)
    images_root.mkdir(parents=True, exist_ok=True)

    scorer = Scorer(args.model) if args.score else None
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    seen = {r[0] for r in conn.execute("SELECT cid FROM images")}
    log.info("%d images already in %s", len(seen), args.db)

    kept = skipped = 0
    started = time.time()
    sem = asyncio.Semaphore(FETCH_CONCURRENCY)

    async def handle(did: str, rkey: str, cid: str) -> None:
        nonlocal kept, skipped
        async with sem:
            raw = await fetch(session, CDN.format(did=did, cid=cid))
        if not raw:
            skipped += 1
            return

        bucket = "random"
        scores = None
        if scorer:
            scores = await asyncio.to_thread(scorer.score, raw)
            if scores is None:
                skipped += 1
                return
            bucket = classify_bucket(scores, args.threshold)
            if args.bucket and bucket != args.bucket:
                skipped += 1
                return

        dest = image_path(images_root, cid)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR IGNORE INTO images (cid, did, rkey, post_uri, path, bytes,"
            " harvested_at, bucket) VALUES (?,?,?,?,?,?,?,?)",
            (cid, did, rkey, f"at://{did}/app.bsky.feed.post/{rkey}",
             str(dest.relative_to(images_root)), len(raw), now, bucket),
        )
        if scores:
            top = max(scores.items(), key=lambda kv: kv[1])
            conn.execute(
                "INSERT OR REPLACE INTO model_scores (image_cid, top_label, top_score,"
                " all_scores, scored_at, model) VALUES (?,?,?,?,?,?)",
                (cid, top[0], top[1], json.dumps(scores), now, args.model),
            )
        conn.commit()
        kept += 1

    tasks: set[asyncio.Task] = set()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=20, connect=5),
        connector=aiohttp.TCPConnector(limit=FETCH_CONCURRENCY),
    ) as session:
        while not stop.is_set() and kept < args.target:
            try:
                async with session.ws_connect(JETSTREAM, heartbeat=30) as ws:
                    log.info("connected to jetstream")
                    async for msg in ws:
                        if stop.is_set() or kept >= args.target:
                            break
                        if msg.type is not aiohttp.WSMsgType.TEXT:
                            continue
                        try:
                            evt = json.loads(msg.data)
                        except json.JSONDecodeError:
                            continue

                        hit = eligible_images(evt)
                        if hit is None:
                            continue
                        # Sample per post, not per image, so a multi-image post
                        # is taken or skipped whole.
                        if random.random() > args.rate:
                            continue

                        did, rkey, cids = hit
                        for cid in cids:
                            if cid in seen:
                                continue
                            seen.add(cid)
                            t = asyncio.create_task(handle(did, rkey, cid))
                            tasks.add(t)
                            t.add_done_callback(tasks.discard)

                        if kept and kept % 100 == 0:
                            rate = kept / max(time.time() - started, 1)
                            log.info("kept=%d skipped=%d (%.1f/s) target=%d",
                                     kept, skipped, rate, args.target)
            except aiohttp.ClientError as e:
                if stop.is_set():
                    break
                log.warning("jetstream disconnected (%s), reconnecting in 5s", e)
                await asyncio.sleep(5)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    conn.close()
    log.info("done: kept=%d skipped=%d in %.0fs", kept, skipped, time.time() - started)


# ── stats / export ────────────────────────────────────────────────────────────

def stats(args: argparse.Namespace) -> None:
    conn = connect(args.db)
    total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    if not total:
        print(f"{args.db} is empty. Run `harvest` first.")
        return

    print(f"{args.db}\n{total:,} images\n")
    print("by bucket")
    for bucket, n in conn.execute(
        "SELECT bucket, COUNT(*) FROM images GROUP BY bucket ORDER BY COUNT(*) DESC"
    ):
        print(f"  {bucket:<14}{n:>8,}")

    rows = conn.execute(
        "SELECT label, source, COUNT(*) FROM labels GROUP BY label, source ORDER BY COUNT(*) DESC"
    ).fetchall()
    print("\nby label" if rows else "\nby label\n  (none yet — harvested images are unlabelled by design)")
    for label, source, n in rows:
        print(f"  {label:<14}{source:<10}{n:>8,}")

    print("\nby split")
    unassigned = conn.execute(
        "SELECT COUNT(*) FROM images WHERE cid NOT IN (SELECT cid FROM splits)"
    ).fetchone()[0]
    for split, n in conn.execute("SELECT split, COUNT(*) FROM splits GROUP BY split"):
        print(f"  {split:<14}{n:>8,}")
    print(f"  {'unassigned':<14}{unassigned:>8,}")

    labelled = conn.execute("SELECT COUNT(DISTINCT cid) FROM labels").fetchone()[0]
    print(f"\n{labelled:,} of {total:,} images have at least one label "
          f"({labelled / total * 100:.1f}%)")
    conn.close()


def verify(args: argparse.Namespace) -> None:
    """Find images already in the manifest whose file is truncated.

    Needed because a bad harvest cannot simply be re-run: cids already in the
    images table are treated as seen and skipped, so the corrupt files would
    persist silently into training.
    """
    conn = connect(args.db)
    conn.row_factory = sqlite3.Row
    root = Path(args.images)

    bad, missing, protected, ok = [], [], [], 0
    for r in conn.execute("SELECT cid, bucket FROM images"):
        path = image_path(root, r["cid"])
        if not path.is_file():
            missing.append(r["cid"])
            continue
        if looks_complete(path.read_bytes()):
            ok += 1
            continue
        # Never silently discard human work: if someone has labelled it, report
        # it and let a person decide.
        human = conn.execute(
            "SELECT 1 FROM labels WHERE cid=? AND source='human' LIMIT 1", (r["cid"],)
        ).fetchone()
        (protected if human else bad).append(r["cid"])

    print(f"complete    {ok:,}")
    print(f"truncated   {len(bad):,}")
    print(f"file gone   {len(missing):,}")
    if protected:
        print(f"truncated but human-labelled  {len(protected):,}  (left alone -- "
              f"re-review these by hand)")
    if not bad and not missing:
        print("\nnothing to repair")
        return

    if not args.repair:
        print("\nre-run with --repair to delete these rows so a fresh harvest "
              "re-fetches them")
        return

    for cid in bad + missing:
        image_path(root, cid).unlink(missing_ok=True)
        for table, col in (("labels", "cid"), ("review_state", "cid"),
                           ("splits", "cid"), ("model_scores", "image_cid"),
                           ("images", "cid")):
            conn.execute(f"DELETE FROM {table} WHERE {col} = ?", (cid,))
    conn.commit()
    conn.close()
    print(f"\nremoved {len(bad) + len(missing):,} rows. Re-run harvest to refetch them.")


def export(args: argparse.Namespace) -> None:
    conn = connect(args.db)
    conn.row_factory = sqlite3.Row

    where, params = [], []
    if args.bucket:
        where.append("i.bucket = ?"); params.append(args.bucket)
    if args.split == "unassigned":
        where.append("i.cid NOT IN (SELECT cid FROM splits)")
    elif args.split:
        where.append("i.cid IN (SELECT cid FROM splits WHERE split = ?)"); params.append(args.split)
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    rows = conn.execute(
        f"SELECT i.*, s.all_scores FROM images i "
        f"LEFT JOIN model_scores s ON s.image_cid = i.cid {clause} "
        f"ORDER BY i.harvested_at LIMIT ?",
        (*params, args.limit),
    ).fetchall()

    out = sys.stdout if args.out == "-" else open(args.out, "w")
    try:
        for r in rows:
            labels = [x[0] for x in conn.execute(
                "SELECT label FROM labels WHERE cid = ? AND source = 'human'", (r["cid"],))]
            out.write(json.dumps({
                "cid": r["cid"],
                "path": r["path"],
                "post_uri": r["post_uri"],
                "bucket": r["bucket"],
                "labels": labels,
                "model_scores": json.loads(r["all_scores"]) if r["all_scores"] else None,
            }) + "\n")
    finally:
        if out is not sys.stdout:
            out.close()
            log.info("wrote %d rows to %s", len(rows), args.out)
    conn.close()


# ── cli ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=default_db(), help="manifest database ($PIPELINE_DB)")
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest", help="sample the firehose into the manifest")
    h.add_argument("--target", type=int, default=1000, help="stop after N new images")
    h.add_argument("--rate", type=float, default=0.05, help="fraction of eligible posts to sample")
    h.add_argument("--images", default=default_images())
    h.add_argument("--score", action="store_true", help="score while harvesting (needs torch)")
    h.add_argument("--model", default=os.getenv("MODEL_NAME", "swin_s3_base_224-xblockm-timm"))
    h.add_argument("--threshold", type=float, default=float(os.getenv("INFERENCE_THRESHOLD", "0.8")))
    h.add_argument("--bucket", choices=["random", "fired", "uncertain", "multi-candidate"],
                   help="keep only this bucket (implies --score)")
    h.set_defaults(fn=harvest)

    s = sub.add_parser("stats", help="what the manifest currently holds")
    s.set_defaults(fn=stats)

    v = sub.add_parser("verify", help="find truncated images already in the manifest")
    v.add_argument("--images", default=default_images())
    v.add_argument("--repair", action="store_true",
                   help="delete the bad rows so a fresh harvest refetches them")
    v.set_defaults(fn=verify)

    e = sub.add_parser("export", help="emit a JSONL review queue")
    e.add_argument("--bucket", choices=["random", "fired", "uncertain", "multi-candidate"])
    e.add_argument("--split", help="a split name, or 'unassigned'")
    e.add_argument("--limit", type=int, default=1000)
    e.add_argument("--out", default="-")
    e.set_defaults(fn=export)

    args = p.parse_args()
    if getattr(args, "bucket", None) and args.cmd == "harvest" and not args.score:
        args.score = True
        log.info("--bucket implies --score")

    if asyncio.iscoroutinefunction(args.fn):
        asyncio.run(args.fn(args))
    else:
        args.fn(args)


if __name__ == "__main__":
    main()
