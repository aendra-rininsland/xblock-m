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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
        ts REAL, images INTEGER, labels_applied INTEGER, duration REAL
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON jobs(ts)")
    conn.commit()
    conn.close()

_init_metrics_db()


def log_metric(images: int, labels_applied: int, duration: float) -> None:
    ts = time.time()
    # JSONL kept for compatibility/backup; SQLite is what the dashboard queries.
    entry = json.dumps({"ts": ts, "images": images, "labels_applied": labels_applied, "duration": duration})
    with open(METRICS_FILE, "a") as f:
        f.write(entry + "\n")
    try:
        conn = sqlite3.connect(METRICS_DB)
        conn.execute("INSERT INTO jobs VALUES (?,?,?,?)", (ts, images, labels_applied, duration))
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
NUM_MODEL_INSTANCES = 1
MODEL_NAME = os.getenv("MODEL_NAME", "swin_s3_base_224-xblockm-timm")

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


logger.info("Loading %d model instances...", NUM_MODEL_INSTANCES)
model_pool: Queue = Queue()
for _ in range(NUM_MODEL_INSTANCES):
    model_pool.put_nowait(create_model_instance())
logger.info("Model instances ready.")


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


async def process_single_image(image_url: str, model, cid: str, top_k: int = 10) -> dict:
    start_time = time.time()
    content = await fetch_image_bytes(image_url)
    if content is None:
        return {"error": "download failed", "url": image_url, "blob_cid": cid, "labels": {}}

    try:
        image = Image.open(BytesIO(content)).convert("RGB")
        cuda_image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(cuda_image)
        probabilities = [float(e) for e in logits.sigmoid().cpu().numpy()[0]]
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

    model = await model_pool.get()
    try:
        results = []
        for data in input_data:
            images = (
                data.get("commit", {})
                    .get("record", {})
                    .get("embed", {})
                    .get("images", [])
            )
            image_urls = [
                (
                    f"https://cdn.bsky.app/img/feed_thumbnail/plain/{data['did']}/{img['image']['ref']['$link']}@jpeg",
                    img["image"]["ref"]["$link"],
                )
                for img in images
            ]

            image_results = []
            if image_urls:
                tasks = [process_single_image(url, model, cid) for url, cid in image_urls]
                image_results = await asyncio.gather(*tasks)
                write_model_scores(image_results)

            results.append({
                "image_results": image_results,
                "commit": data.get("commit", {}),
                "did": data["did"],
            })
    finally:
        await model_pool.put(model)

    labels_applied = 0
    for result in results:
        for image_result in result["image_results"]:
            for label, score in image_result.get("labels", {}).items():
                if label != "negative" and float(score) > THRESHOLD:
                    await create_label(result)
                    labels_applied += 1

    duration = time.time() - start_time
    total_images = sum(len(r["image_results"]) for r in results)
    log_metric(total_images, labels_applied, duration)
    logger.info("job done images=%d labels=%d duration=%.2fs", total_images, labels_applied, duration)

    return results if len(results) > 1 else results[0]


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    await auth_client()

    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        logger.info("Signal %s received, shutting down...", sig)
        shutdown_event.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("Starting BullMQ worker...")
    worker = Worker(
        "xblock",
        process_request,
        {"connection": os.environ["REDIS_CONNECTION_STRING"], "concurrency": NUM_WORKERS},
    )

    await shutdown_event.wait()

    logger.info("Closing worker...")
    if _http_session and not _http_session.closed:
        await _http_session.close()
    await worker.close(force=True)
    logger.info("Worker shut down cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
