#!/usr/bin/env python
import asyncio
from asyncio import Queue
import datetime
import json
import logging
import os
import signal
import sqlite3
import time


import aiohttp
import torch
import torchvision.transforms as T
from bullmq import Worker
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from io import BytesIO
from PIL import Image
from safetensors.torch import load_file
from timm import create_model

from constants import THRESHOLD
from moderate import auth_client, create_label

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
METRICS_FILE = os.path.join(LOG_DIR, "metrics.jsonl")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("xblock")


METRICS_DB = os.path.join(LOG_DIR, "metrics.db")

def _init_metrics_db() -> None:
    conn = sqlite3.connect(METRICS_DB)
    # WAL lets the dashboard read while the worker writes. Now that jobs complete
    # orders of magnitude faster, the default rollback journal would make readers
    # and the writer block each other ("database is locked").
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
        ts REAL, images INTEGER, labels_applied INTEGER, duration REAL,
        errors INTEGER DEFAULT 0
    )""")
    # Databases created before the errors column exist in the wild, and
    # CREATE TABLE IF NOT EXISTS will not add it to them.
    columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    if "errors" not in columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN errors INTEGER DEFAULT 0")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON jobs(ts)")
    conn.commit()
    conn.close()

_init_metrics_db()


def log_metric(images: int, labels_applied: int, duration: float, errors: int = 0) -> None:
    ts = time.time()
    # JSONL kept for compatibility/backup; SQLite is what the dashboard queries.
    entry = json.dumps({"ts": ts, "images": images, "labels_applied": labels_applied,
                        "duration": duration, "errors": errors})
    with open(METRICS_FILE, "a") as f:
        f.write(entry + "\n")
    try:
        conn = sqlite3.connect(METRICS_DB, timeout=5)
        # Named columns rather than positional, so a future migration cannot
        # silently shift the values.
        conn.execute(
            "INSERT INTO jobs (ts, images, labels_applied, duration, errors) VALUES (?,?,?,?,?)",
            (ts, images, labels_applied, duration, errors),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("metrics db write failed: %s", e)


# ── Pipeline DB ───────────────────────────────────────────────────────────────
# Optional: set PIPELINE_DB to the absolute path of the retraining pipeline's
# SQLite database so that model scores are available to the auto-labeller.
# Example (WSL2): /mnt/d/Claude/Projects/Projects/XBlock Training Pipeline/data/pipeline.db
PIPELINE_DB = os.getenv("PIPELINE_DB", "")
if PIPELINE_DB:
    if os.path.exists(PIPELINE_DB):
        logger.info("Pipeline DB found at %s — model scores will be persisted.", PIPELINE_DB)
    else:
        logger.warning("PIPELINE_DB is set to %s but file does not exist yet. "
                       "Scores will be written once the pipeline has run at least once.", PIPELINE_DB)


def write_model_scores(image_results: list) -> None:
    """Persist model confidence scores to the pipeline DB for the auto-labeller."""
    if not PIPELINE_DB:
        return
    try:
        conn = sqlite3.connect(PIPELINE_DB)
        try:
            for r in image_results:
                if r.get("error") or not r.get("labels"):
                    continue
                labels: dict = r["labels"]
                if not labels:
                    continue
                top_label = next(iter(labels))
                top_score = labels[top_label]
                scored_at = datetime.datetime.utcnow().isoformat() + "Z"
                conn.execute(
                    """INSERT OR REPLACE INTO model_scores
                         (image_cid, top_label, top_score, all_scores, scored_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (r["blob_cid"], top_label, float(top_score),
                     json.dumps(labels), scored_at),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.error("Failed to write model scores to pipeline DB: %s", e)


# ── Model loading ─────────────────────────────────────────────────────────────

torch.set_num_threads(1)

NUM_WORKERS = 50
MODEL_NAME = os.getenv("MODEL_NAME", "swin_s3_base_224-xblockm-timm")

# Micro-batching. Images arrive one or two at a time from independent jobs, but a
# GPU wants them in a batch. Requests are collected for up to INFER_BATCH_WAIT_MS
# (or until INFER_BATCH_SIZE is reached) and run as a single forward pass.
INFER_BATCH_SIZE = int(os.getenv("INFER_BATCH_SIZE", "16"))
INFER_BATCH_WAIT_MS = int(os.getenv("INFER_BATCH_WAIT_MS", "25"))

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using device: %s", device)

model_id = f"howdyaendra/{MODEL_NAME}"
cache_dir = "./models"

model_weights_path = hf_hub_download(repo_id=model_id, filename="model.safetensors", cache_dir=cache_dir)
config_path = hf_hub_download(repo_id=model_id, filename="config.json", cache_dir=cache_dir)

with open(config_path) as f:
    config = json.load(f)

num_classes = config.get("num_classes", 13)
model_name = "swin_s3_base_224"

img_size = (224, 224)
transform = T.Compose([
    T.Resize(img_size),
    T.CenterCrop(img_size),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


def create_model_instance():
    model = create_model(model_name, num_classes=num_classes, pretrained=False)
    model.to(device)
    state_dict = load_file(model_weights_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model


logger.info("Loading model...")
# A single instance is all that is needed: the batcher below is the only caller
# and it runs one forward pass at a time, so there is nothing to contend over.
model = create_model_instance()
logger.info("Model ready. batch_size=%d batch_wait=%dms", INFER_BATCH_SIZE, INFER_BATCH_WAIT_MS)


# ── Inference batcher ─────────────────────────────────────────────────────────
# Callers submit a single pre-processed tensor and await a future. One background
# task collects them into batches and runs the forward pass. Crucially, nothing
# here is held during image download — only the ~15ms of GPU work is serialised.

_infer_queue: Queue | None = None
_batcher_task: asyncio.Task | None = None


def _decode(content: bytes) -> torch.Tensor:
    """Blocking decode + preprocess. Runs in a worker thread."""
    image = Image.open(BytesIO(content)).convert("RGB")
    return transform(image)


def _run_batch(tensors: list[torch.Tensor]):
    """Blocking forward pass over a stacked batch. Runs in a worker thread."""
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        logits = model(batch)
    return logits.sigmoid().cpu().numpy()


def _fail_waiters(items: list, exc: BaseException) -> None:
    """Resolve every future in `items`. A waiter left unresolved hangs its job."""
    for _, future in items:
        if not future.done():
            future.set_exception(exc)


async def _batch_loop() -> None:
    assert _infer_queue is not None
    loop = asyncio.get_running_loop()
    while True:
        items: list = []
        try:
            # Block until there is at least one request, then top up the batch
            # until it is full or the wait window expires, whichever is first.
            items.append(await _infer_queue.get())
            deadline = loop.time() + INFER_BATCH_WAIT_MS / 1000
            while len(items) < INFER_BATCH_SIZE:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    items.append(await asyncio.wait_for(_infer_queue.get(), remaining))
                except asyncio.TimeoutError:
                    break

            pending = [(t, f) for t, f in items if not f.done()]
            if not pending:
                continue

            probs = await asyncio.to_thread(_run_batch, [t for t, _ in pending])
            for (_, f), row in zip(pending, probs):
                if not f.done():
                    f.set_result(row)

        except asyncio.CancelledError:
            # Shutdown. CancelledError is a BaseException, so without this the
            # futures for any batch in flight would never resolve and their jobs
            # would await them forever. Raise a plain exception at the waiters so
            # process_single_image can catch it and return an error result.
            _fail_waiters(items, RuntimeError("inference batcher stopped"))
            raise
        except Exception as e:
            logger.error("Batch inference failed (n=%d): %s", len(items), e)
            _fail_waiters(items, e)


async def _infer(tensor: torch.Tensor):
    """Submit one tensor for batched inference and await its scores."""
    if _infer_queue is None:
        raise RuntimeError("inference batcher is not running")
    future = asyncio.get_running_loop().create_future()
    await _infer_queue.put((tensor, future))
    return await future


# ── HTTP session ──────────────────────────────────────────────────────────────
# One shared session for the lifetime of the process, created lazily once the
# event loop is running.

_http_session: aiohttp.ClientSession | None = None
_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=20, connect=5)


def get_http_session() -> aiohttp.ClientSession:
    global _http_session
    if _http_session is None or _http_session.closed:
        # Keep the connection pool tight — we don't need more open sockets than
        # the BullMQ concurrency level, and a large pool causes network pressure
        # that competes with other apps on the machine.
        connector = aiohttp.TCPConnector(limit=NUM_WORKERS, ttl_dns_cache=300)
        _http_session = aiohttp.ClientSession(timeout=_HTTP_TIMEOUT, connector=connector)
    return _http_session


# ── Image processing ──────────────────────────────────────────────────────────

_IMAGE_RETRIES = 3
_IMAGE_RETRY_BASE = 1.0  # seconds


async def fetch_image_bytes(url: str) -> bytes | None:
    """Download with retries and exponential backoff. Returns None on permanent failure."""
    session = get_http_session()
    for attempt in range(_IMAGE_RETRIES):
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.read()
                if resp.status < 500:
                    # 4xx — not a transient error, don't retry
                    logger.warning("Image fetch %s returned HTTP %d", url, resp.status)
                    return None
                raise aiohttp.ClientResponseError(resp.request_info, resp.history, status=resp.status)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == _IMAGE_RETRIES - 1:
                logger.error("Image fetch %s failed after %d attempts: %s", url, _IMAGE_RETRIES, e)
                return None
            wait = _IMAGE_RETRY_BASE * (2 ** attempt)
            logger.warning("Image fetch %s attempt %d failed (%s), retrying in %.1fs...", url, attempt + 1, e, wait)
            await asyncio.sleep(wait)
    return None


async def process_single_image(image_url: str, cid: str, top_k: int = 10) -> dict:
    start_time = time.time()
    content = await fetch_image_bytes(image_url)
    if content is None:
        return {"error": "download failed", "url": image_url, "blob_cid": cid, "labels": {}}

    try:
        # Decode off the event loop — PIL and the torchvision transform are
        # blocking CPU work and would otherwise stall every concurrent download.
        tensor = await asyncio.to_thread(_decode, content)
        probabilities = [float(e) for e in await _infer(tensor)]
        label_prob_pairs = sorted(zip(config["label_names"], probabilities), key=lambda x: x[1], reverse=True)
        return {
            "image_url": image_url,
            "blob_cid": cid,
            "labels": dict(label_prob_pairs[:top_k]),
            "time": time.time() - start_time,
        }
    except Exception as e:
        logger.error("Image inference failed for %s: %s", image_url, e)
        return {"error": str(e), "url": image_url, "blob_cid": cid, "labels": {}}


# ── Job processing ────────────────────────────────────────────────────────────

async def process_request(job, token):
    start_time = time.time()
    input_data = job.data
    if isinstance(input_data, dict):
        input_data = [input_data]

    # Flatten every image in the job so they all download concurrently, then
    # regroup by post. Nothing is serialised here.
    plan: list[tuple[int, str, str]] = []
    for idx, data in enumerate(input_data):
        images = (
            data.get("commit", {})
                .get("record", {})
                .get("embed", {})
                .get("images", [])
        )
        for img in images:
            cid = img["image"]["ref"]["$link"]
            url = f"https://cdn.bsky.app/img/feed_thumbnail/plain/{data['did']}/{cid}@jpeg"
            plan.append((idx, url, cid))

    flat = await asyncio.gather(*(process_single_image(url, cid) for _, url, cid in plan))

    grouped: dict[int, list] = {idx: [] for idx in range(len(input_data))}
    for (idx, _, _), image_result in zip(plan, flat):
        grouped[idx].append(image_result)

    results = []
    for idx, data in enumerate(input_data):
        image_results = grouped[idx]
        if image_results:
            write_model_scores(image_results)
        results.append({
            "image_results": image_results,
            "commit": data.get("commit", {}),
            "did": data["did"],
        })

    labels_applied = 0
    errors = 0
    for result in results:
        hits = 0
        for image_result in result["image_results"]:
            if image_result.get("error"):
                errors += 1
            for label, score in image_result.get("labels", {}).items():
                if label != "negative" and float(score) >= THRESHOLD:
                    hits += 1
        # One event per post. _emit_label re-derives the full label set from the
        # result itself, so calling it once per matching label emitted the same
        # event several times over.
        if hits:
            await create_label(result)
            labels_applied += hits

    duration = time.time() - start_time
    total_images = sum(len(r["image_results"]) for r in results)
    log_metric(total_images, labels_applied, duration, errors)
    logger.info("job done images=%d labels=%d errors=%d duration=%.2fs",
                total_images, labels_applied, errors, duration)

    return results if len(results) > 1 else results[0]


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    global _infer_queue, _batcher_task

    await auth_client()

    # Created here rather than at import time so they bind to the running loop.
    _infer_queue = Queue()
    _batcher_task = asyncio.create_task(_batch_loop())

    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_shutdown(sig_name: str) -> None:
        logger.info("Signal %s received, shutting down...", sig_name)
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            # add_signal_handler runs the callback on the loop. signal.signal
            # would run it between bytecodes instead, where touching loop state
            # is not safe.
            loop.add_signal_handler(sig, request_shutdown, sig.name)
        except NotImplementedError:
            # Windows event loops have no add_signal_handler.
            signal.signal(
                sig,
                lambda s, f: loop.call_soon_threadsafe(request_shutdown, signal.Signals(s).name),
            )

    logger.info("Starting BullMQ worker...")
    worker = Worker(
        "xblock",
        process_request,
        {"connection": os.environ["REDIS_CONNECTION_STRING"], "concurrency": NUM_WORKERS},
    )

    await shutdown_event.wait()

    logger.info("Closing worker...")
    await worker.close(force=True)
    if _batcher_task:
        _batcher_task.cancel()
        try:
            await _batcher_task
        except asyncio.CancelledError:
            pass
    if _http_session and not _http_session.closed:
        await _http_session.close()
    logger.info("Worker shut down cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
