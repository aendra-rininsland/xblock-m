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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("harvest")

JETSTREAM = os.getenv(
    "JETSTREAM_URL",
    "wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post",
)
CDN = "https://cdn.bsky.app/img/feed_thumbnail/plain/{did}/{cid}@jpeg"

DEFAULT_DB = os.getenv("PIPELINE_DB") or str(Path(__file__).parent / "data" / "pipeline.db")
UNCERTAIN_BAND = (0.35, 0.75)
FETCH_CONCURRENCY = 16
MAX_IMAGE_BYTES = 8 * 1024 * 1024


# ── schema ────────────────────────────────────────────────────────────────────

SCHEMA = """
PRAGMA journal_mode=WAL;

-- One row per distinct image. cid is a content hash, so this deduplicates
-- byte-identical re-uploads for free and makes harvests resumable.
CREATE TABLE IF NOT EXISTS images (
    cid          TEXT PRIMARY KEY,
    did          TEXT NOT NULL,
    rkey         TEXT NOT NULL,
    post_uri     TEXT NOT NULL,
    path         TEXT NOT NULL,
    bytes        INTEGER,
    harvested_at TEXT NOT NULL,
    bucket       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_images_bucket ON images(bucket);

-- The point of the exercise: an image carries a SET of labels, not one.
-- `source` records provenance, so a human decision can outrank a model guess
-- and an Ozone appeal can be told apart from a hand review.
CREATE TABLE IF NOT EXISTS labels (
    cid        TEXT NOT NULL,
    label      TEXT NOT NULL,
    source     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (cid, label, source)
);
CREATE INDEX IF NOT EXISTS idx_labels_cid ON labels(cid);

-- Kept separate from labels so that relabelling an image never silently moves
-- it between train and eval. A frozen eval set only stays frozen if its
-- membership lives somewhere relabelling does not touch.
CREATE TABLE IF NOT EXISTS splits (
    cid         TEXT PRIMARY KEY,
    split       TEXT NOT NULL,
    assigned_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_splits_split ON splits(split);

-- Shape matches what processor/worker.py already writes, so the two can share a
-- database. Note image_cid is the sole primary key: one score row per image, not
-- one per (image, model). Storing scores from several models -- which shadow
-- comparison would want -- needs a migration first.
CREATE TABLE IF NOT EXISTS model_scores (
    image_cid  TEXT PRIMARY KEY,
    top_label  TEXT,
    top_score  REAL,
    all_scores TEXT,
    scored_at  TEXT,
    model      TEXT
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10)
    conn.executescript(SCHEMA)
    # Older databases predate `model`; CREATE TABLE IF NOT EXISTS will not add it.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(model_scores)")}
    if "model" not in cols:
        conn.execute("ALTER TABLE model_scores ADD COLUMN model TEXT")
    conn.commit()
    return conn


def image_path(root: Path, cid: str) -> Path:
    """Shard by the first two characters so no directory holds 10k+ files."""
    return root / cid[:2] / f"{cid}.jpeg"


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
    """Which stratum this image belongs to, given the current model's opinion."""
    non_neg = [v for k, v in scores.items() if k != "negative"]
    top = max(non_neg) if non_neg else 0.0
    if top >= threshold:
        return "fired"
    if UNCERTAIN_BAND[0] <= top <= UNCERTAIN_BAND[1]:
        return "uncertain"
    return "random"


# ── harvest ───────────────────────────────────────────────────────────────────

async def fetch(session: aiohttp.ClientSession, url: str) -> bytes | None:
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                return None
            if resp.content_length and resp.content_length > MAX_IMAGE_BYTES:
                log.warning("skipping oversized image (%s bytes)", resp.content_length)
                return None
            return await resp.content.read(MAX_IMAGE_BYTES + 1)
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
    p.add_argument("--db", default=DEFAULT_DB, help=f"manifest database (default: {DEFAULT_DB})")
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest", help="sample the firehose into the manifest")
    h.add_argument("--target", type=int, default=1000, help="stop after N new images")
    h.add_argument("--rate", type=float, default=0.05, help="fraction of eligible posts to sample")
    h.add_argument("--images", default=str(Path(__file__).parent / "data" / "images"))
    h.add_argument("--score", action="store_true", help="score while harvesting (needs torch)")
    h.add_argument("--model", default=os.getenv("MODEL_NAME", "swin_s3_base_224-xblockm-timm"))
    h.add_argument("--threshold", type=float, default=float(os.getenv("INFERENCE_THRESHOLD", "0.8")))
    h.add_argument("--bucket", choices=["random", "fired", "uncertain"],
                   help="keep only this bucket (implies --score)")
    h.set_defaults(fn=harvest)

    s = sub.add_parser("stats", help="what the manifest currently holds")
    s.set_defaults(fn=stats)

    e = sub.add_parser("export", help="emit a JSONL review queue")
    e.add_argument("--bucket", choices=["random", "fired", "uncertain"])
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
