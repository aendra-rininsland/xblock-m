"""
Manifest schema and class list, shared by the harvester and the review UI.

The central design decision here is that an image carries a *set* of labels. The
training corpus is an ImageFolder tree -- one directory per class, one label per
image -- which structurally cannot express a co-occurrence, so the
nested-screenshot case (a Bluesky post quoting a Twitter screenshot) is
unlearnable no matter how the model is trained. This schema is the fix.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

# Model classes as of the altright -> truthsocial rename and the news drop.
# Must stay in step with LABEL_VALUES in processor/moderate.py, which maps these
# to the values actually published to subscribers, and with label_names in the
# training notebook. Ordered by corpus frequency so the review UI can give the
# commonest classes the easiest keys.
NEGATIVE = "negative"
PLATFORM_CLASSES = [
    "twitter", "bluesky", "threads", "discord", "facebook", "instagram",
    "reddit", "truthsocial", "tumblr", "fediverse", "ngl",
]
CLASSES = PLATFORM_CLASSES + [NEGATIVE]

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

-- Review progress, kept apart from labels so that "confirmed negative" and
-- "not looked at yet" are distinguishable. Without this an image with no labels
-- is ambiguous, and a 3,000-image sweep cannot be resumed.
CREATE TABLE IF NOT EXISTS review_state (
    cid        TEXT PRIMARY KEY,
    state      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_review_state ON review_state(state);

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

# Review states.
PENDING = "pending"            # never looked at
NEEDS_DETAIL = "needs-detail"  # grid sweep said "this is a screenshot" -- needs classes
DONE = "done"                  # labelled, including confirmed negative
SKIPPED = "skipped"            # deliberately passed over (unclear, NSFW, broken)


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
