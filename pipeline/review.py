#!/usr/bin/env python
"""
Local review UI for the harvest manifest.

Two modes, because the work has two very different shapes:

  sweep   A grid of thumbnails. Click the few that ARE screenshots; everything
          left unclicked is confirmed negative on submit. This is how ~3,000
          negatives get labelled in well under an hour -- the corpus is short of
          negatives by roughly that much, and it is the largest remaining driver
          of production false positives.

  detail  One image at a time, multi-select classes by keyboard. For images the
          sweep flagged, and for the `fired` and `uncertain` buckets where the
          judgement actually matters.

Nothing is auto-labelled. The sweep infers `negative` only from an explicit
human submit of that page.

    pip install -r requirements.txt fastapi 'uvicorn[standard]'
    python review.py --db data/pipeline.db --images data/images
    # http://127.0.0.1:8081
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from manifest import (CLASSES, DONE, IMPORTED, NEEDS_DETAIL, NEGATIVE, PENDING,
                      PLATFORM_CLASSES, SKIPPED, connect, image_path)

app = FastAPI(title="xblock review")

DB_PATH = ""
IMAGES_ROOT = Path()
HUMAN = "human"

# Bounded so a long session cannot grow without limit. Undo is a convenience for
# slips during a sweep, not an audit log -- the labels table is that.
_undo: deque[list[dict]] = deque(maxlen=200)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def db() -> sqlite3.Connection:
    conn = connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _snapshot(conn: sqlite3.Connection, cids: list[str]) -> list[dict]:
    """Capture enough to reverse a write, before making it."""
    snap = []
    for cid in cids:
        labels = [r["label"] for r in conn.execute(
            "SELECT label FROM labels WHERE cid=? AND source=?", (cid, HUMAN))]
        row = conn.execute("SELECT state FROM review_state WHERE cid=?", (cid,)).fetchone()
        snap.append({"cid": cid, "labels": labels, "state": row["state"] if row else None})
    return snap


def _apply(conn: sqlite3.Connection, cid: str, labels: list[str], state: str) -> None:
    conn.execute("DELETE FROM labels WHERE cid=? AND source=?", (cid, HUMAN))
    for label in labels:
        conn.execute(
            "INSERT OR REPLACE INTO labels (cid,label,source,created_at) VALUES (?,?,?,?)",
            (cid, label, HUMAN, now()))
    conn.execute(
        "INSERT INTO review_state (cid,state,updated_at) VALUES (?,?,?) "
        "ON CONFLICT(cid) DO UPDATE SET state=excluded.state, updated_at=excluded.updated_at",
        (cid, state, now()))


# ── api ───────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "review.html").read_text()


@app.get("/api/config")
async def config():
    return {"classes": CLASSES, "platforms": PLATFORM_CLASSES, "negative": NEGATIVE}


@app.get("/api/queue")
async def queue(mode: str = "sweep", bucket: str = "", limit: int = 60):
    """Next batch to review.

    sweep  -> untouched images, newest harvest first
    detail -> whatever the sweep flagged, then anything still pending
    """
    conn = db()
    # Imported ImageFolder rows already carry a label, so they stay out of the
    # default queue -- but selecting their bucket explicitly is how the
    # re-review pass surfaces them.
    hidden = [DONE, SKIPPED] if bucket else [DONE, SKIPPED, IMPORTED]
    where = [f"i.cid NOT IN (SELECT cid FROM review_state WHERE state IN "
             f"({','.join('?' * len(hidden))}))"]
    params: list = [*hidden]
    if bucket:
        where.append("i.bucket = ?")
        params.append(bucket)
    if mode == "detail":
        order = ("CASE WHEN rs.state = '%s' THEN 0 ELSE 1 END, i.harvested_at DESC"
                 % NEEDS_DETAIL)
    else:
        where.append("(rs.state IS NULL OR rs.state = ?)")
        params.append(PENDING)
        order = "i.harvested_at DESC"

    rows = conn.execute(
        f"SELECT i.cid, i.bucket, i.post_uri, s.all_scores, rs.state "
        f"FROM images i "
        f"LEFT JOIN model_scores s ON s.image_cid = i.cid "
        f"LEFT JOIN review_state rs ON rs.cid = i.cid "
        f"WHERE {' AND '.join(where)} ORDER BY {order} LIMIT ?",
        (*params, limit),
    ).fetchall()

    items = []
    for r in rows:
        scores = json.loads(r["all_scores"]) if r["all_scores"] else None
        if scores:
            scores = dict(sorted(scores.items(), key=lambda kv: -kv[1])[:4])
        # Existing labels come back so the UI can pre-select them. Review of an
        # already-labelled image is additive -- confirming a second platform --
        # not re-entry from scratch.
        labels = [x[0] for x in conn.execute(
            "SELECT label FROM labels WHERE cid = ?", (r["cid"],))]
        items.append({"cid": r["cid"], "bucket": r["bucket"], "post_uri": r["post_uri"],
                      "state": r["state"], "scores": scores, "labels": sorted(set(labels))})
    conn.close()
    return {"items": items}


@app.get("/api/stats")
async def stats():
    conn = db()
    total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    by_state = {r["state"]: r["n"] for r in conn.execute(
        "SELECT state, COUNT(*) n FROM review_state GROUP BY state")}
    reviewed = by_state.get(DONE, 0)
    by_label = {r["label"]: r["n"] for r in conn.execute(
        "SELECT label, COUNT(*) n FROM labels WHERE source=? GROUP BY label ORDER BY n DESC",
        (HUMAN,))}
    negatives = by_label.get(NEGATIVE, 0)
    positives = sum(v for k, v in by_label.items() if k != NEGATIVE)
    conn.close()
    return {"total": total, "reviewed": reviewed, "pending": total - reviewed,
            "by_state": by_state, "by_label": by_label,
            "negatives": negatives, "positives": positives,
            "ratio": round(negatives / positives, 2) if positives else None}


@app.get("/api/image/{cid}")
async def image(cid: str):
    # cid comes from the URL; refuse anything that is not a bare content hash so
    # it cannot be walked into a path outside the image root.
    if not cid.isalnum():
        raise HTTPException(400, "bad cid")
    path = image_path(IMAGES_ROOT, cid)
    if not path.is_file():
        raise HTTPException(404, "not harvested")
    return FileResponse(path, media_type="image/jpeg")


class SweepBody(BaseModel):
    negative: list[str] = []   # unclicked -> confirmed not a screenshot
    flagged: list[str] = []    # clicked   -> is a screenshot, needs classes


@app.post("/api/sweep")
async def sweep(body: SweepBody):
    conn = db()
    _undo.append(_snapshot(conn, body.negative + body.flagged))
    for cid in body.negative:
        _apply(conn, cid, [NEGATIVE], DONE)
    for cid in body.flagged:
        # No labels yet -- being a screenshot is not itself a label.
        _apply(conn, cid, [], NEEDS_DETAIL)
    conn.commit()
    conn.close()
    return {"negative": len(body.negative), "flagged": len(body.flagged)}


class LabelBody(BaseModel):
    cid: str
    labels: list[str] = []
    skip: bool = False


@app.post("/api/label")
async def label(body: LabelBody):
    unknown = set(body.labels) - set(CLASSES)
    if unknown:
        raise HTTPException(400, f"unknown classes: {sorted(unknown)}")
    if NEGATIVE in body.labels and len(body.labels) > 1:
        raise HTTPException(400, "negative is exclusive: an image is either not a "
                                 "screenshot, or it is one of something")
    conn = db()
    _undo.append(_snapshot(conn, [body.cid]))
    _apply(conn, body.cid, body.labels, SKIPPED if body.skip else DONE)
    conn.commit()
    conn.close()
    return {"ok": True}


@app.post("/api/undo")
async def undo():
    if not _undo:
        return JSONResponse({"ok": False, "reason": "nothing to undo"}, status_code=409)
    conn = db()
    for entry in _undo.pop():
        conn.execute("DELETE FROM labels WHERE cid=? AND source=?", (entry["cid"], HUMAN))
        for label in entry["labels"]:
            conn.execute(
                "INSERT OR REPLACE INTO labels (cid,label,source,created_at) VALUES (?,?,?,?)",
                (entry["cid"], label, HUMAN, now()))
        if entry["state"] is None:
            conn.execute("DELETE FROM review_state WHERE cid=?", (entry["cid"],))
        else:
            conn.execute(
                "INSERT INTO review_state (cid,state,updated_at) VALUES (?,?,?) "
                "ON CONFLICT(cid) DO UPDATE SET state=excluded.state",
                (entry["cid"], entry["state"], now()))
    conn.commit()
    conn.close()
    return {"ok": True}


def main() -> None:
    global DB_PATH, IMAGES_ROOT
    here = Path(__file__).parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=str(here / "data" / "pipeline.db"))
    p.add_argument("--images", default=str(here / "data" / "images"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8081)
    args = p.parse_args()

    DB_PATH = args.db
    IMAGES_ROOT = Path(args.images)
    connect(DB_PATH).close()

    import uvicorn
    print(f"reviewing {DB_PATH}  ->  http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
