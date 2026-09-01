#!/usr/bin/env python
"""
Import moderation signal from the Ozone Postgres database into the manifest.

What Ozone is good for
----------------------
It holds the moderation record, not the images -- but `moderation_event` carries
`subjectBlobCids`, so the images are recoverable from the CDN by CID. What it has
that the firehose cannot give:

  reports        user-submitted, the actual false-positive signal
  appeals        a user disputing a label -- the strongest FP evidence there is
  negations      a human removing a label the model applied: a confirmed FP with
                 a moderator's judgement attached
  model labels   every image the labeller has ever fired on

What it is bad for
------------------
Everything in it is either something the model already fired on or something a
user complained about. It is not a sample of the firehose, and it contains almost
no negatives -- which is the corpus's largest gap. Use harvest.py for those.

More importantly, model labels here are the model's OWN predictions. Importing
them as training labels trains the model on its own output, which entrenches the
current decision boundary instead of correcting it -- and drift is precisely a
decision-boundary problem. So they arrive as `source='ozone-model'` and go to
review as candidates. Nothing from Ozone is written as `source='human'` except an
actual human moderator's action.

Distinguishing who did what
---------------------------
  action='...#modEventReport'   a report, never a label
  action='...#modEventLabel'    a label applied or negated
  createdBy = the labeller DID  the model; anything else is a human moderator
  modTool.meta.isAutomated      Ozone's own field for this, set by moderate.py

Older events predate modTool, so `createdBy` is the discriminator that works
across the whole history.

    export OZONE_DB_URL=postgres://user:pass@host/ozone
    python import_ozone.py --bot-did did:plc:...            # dry run
    python import_ozone.py --bot-did did:plc:... --apply --since 2025-01-01
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "processor"))

from label_values import model_class  # noqa: E402
from manifest import (IMPORTED, NEEDS_DETAIL, connect, default_db,
                      default_images, image_path)

CDN = "https://cdn.bsky.app/img/feed_thumbnail/plain/{did}/{cid}@jpeg"

LABEL_EVENT = "tools.ozone.moderation.defs#modEventLabel"
REPORT_EVENT = "tools.ozone.moderation.defs#modEventReport"
APPEAL_REASONS = ("com.atproto.moderation.defs#reasonAppeal",
                  "tools.ozone.report.defs#reasonAppeal")

# Buckets, so the review UI can work each kind of signal separately.
B_MODEL = "ozone-model"        # the labeller fired; unverified
B_HUMAN = "ozone-human"        # a moderator applied this by hand
B_FP = "ozone-false-positive"  # a human removed a label the model applied
B_REPORT = "ozone-reported"    # someone reported it
B_APPEAL = "ozone-appeal"      # someone appealed a label


def as_text(v):
    """Text columns normally arrive as str, but a cluster created with SQL_ASCII
    encoding makes psycopg hand back bytes instead. Decode rather than failing
    with a TypeError several frames deep in a parser."""
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    return v


def split_vals(raw) -> list[str]:
    """createLabelVals/negateLabelVals are varchar, not arrays -- comma-joined."""
    raw = as_text(raw)
    if not raw:
        return []
    return [v.strip() for v in raw.split(",") if v.strip()]


def blob_cids(raw) -> list[str]:
    """subjectBlobCids is jsonb; psycopg may hand back a list or a string."""
    if not raw:
        return []
    raw = as_text(raw)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    return [c for c in raw if isinstance(c, str)] if isinstance(raw, list) else []


def report_type(meta) -> str | None:
    """reportType lives in moderation_event.meta (jsonb). The `report` table
    denormalises it, but that table only exists in Ozone from Feb 2026, so read
    it from the event -- which works on every version."""
    meta = as_text(meta)
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            return None
    return meta.get("reportType") if isinstance(meta, dict) else None


def classes_from(vals: list[str]) -> list[str]:
    """Published label values -> current model classes, dropping unknowns."""
    return [c for c in (model_class(v) for v in vals) if c]


QUERIES: dict[str, str] = {
    # Columns are camelCase and must be quoted. subjectBlobCids was added in
    # Feb 2024; earlier events have NULL and their images are not recoverable
    # from the event alone.
    B_MODEL: """
        SELECT "subjectUri", "subjectDid", "subjectBlobCids", "createLabelVals",
               "createdAt"
          FROM moderation_event
         WHERE action = %(label_event)s
           AND "createdBy" = %(bot)s
           AND "createLabelVals" IS NOT NULL AND "createLabelVals" <> ''
           AND "createdAt" >= %(since)s
         ORDER BY "createdAt" DESC LIMIT %(limit)s
    """,
    B_HUMAN: """
        SELECT "subjectUri", "subjectDid", "subjectBlobCids", "createLabelVals",
               "createdAt"
          FROM moderation_event
         WHERE action = %(label_event)s
           AND "createdBy" <> %(bot)s
           AND "createLabelVals" IS NOT NULL AND "createLabelVals" <> ''
           AND "createdAt" >= %(since)s
         ORDER BY "createdAt" DESC LIMIT %(limit)s
    """,
    # The gold signal: a human negated a label, on a subject the model had
    # labelled. The negation says the label was wrong; it does not say what is
    # right, so these import with NO label and go to review.
    B_FP: """
        SELECT n."subjectUri", n."subjectDid", m."subjectBlobCids",
               n."negateLabelVals", n."createdAt"
          FROM moderation_event n
          JOIN moderation_event m
            ON m."subjectUri" = n."subjectUri"
           AND m.action = %(label_event)s
           AND m."createdBy" = %(bot)s
           AND m."createLabelVals" IS NOT NULL
         WHERE n.action = %(label_event)s
           AND n."createdBy" <> %(bot)s
           AND n."negateLabelVals" IS NOT NULL AND n."negateLabelVals" <> ''
           AND n."createdAt" >= %(since)s
         ORDER BY n."createdAt" DESC LIMIT %(limit)s
    """,
    B_REPORT: """
        SELECT "subjectUri", "subjectDid", "subjectBlobCids", meta, "createdAt"
          FROM moderation_event
         WHERE action = %(report_event)s
           AND "subjectUri" IS NOT NULL
           AND "createdAt" >= %(since)s
         ORDER BY "createdAt" DESC LIMIT %(limit)s
    """,
}


def fetch_image(session, did: str, cid: str) -> bytes | None:
    try:
        r = session.get(CDN.format(did=did, cid=cid), timeout=20)
        return r.content if r.status_code == 200 and r.content else None
    except Exception:
        return None


def main() -> None:
    here = Path(__file__).parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=default_db())
    p.add_argument("--images", default=default_images())
    p.add_argument("--dsn", default=os.getenv("OZONE_DB_URL"))
    p.add_argument("--bot-did", default=os.getenv("OZONE_BOT_DID"),
                   help="the labeller's DID -- the discriminator for model vs human")
    p.add_argument("--since", default="2024-02-01",
                   help="subjectBlobCids was added Feb 2024; earlier events have no images")
    p.add_argument("--limit", type=int, default=20000, help="per signal kind")
    p.add_argument("--kinds", default=",".join(QUERIES),
                   help="comma-separated subset of: " + ", ".join(QUERIES))
    p.add_argument("--dry-run", action="store_true",
                   help="report without writing (this is the default)")
    p.add_argument("--apply", action="store_true", help="write (default is a dry run)")
    args = p.parse_args()

    if args.dry_run and args.apply:
        sys.exit("--dry-run and --apply are contradictory; a dry run is the default")

    if not args.dsn:
        sys.exit("set OZONE_DB_URL or pass --dsn")
    if not args.bot_did:
        sys.exit("--bot-did is required: it is what separates the model's own "
                 "labels from a human moderator's")
    try:
        import psycopg
    except ImportError:
        sys.exit("pip install 'psycopg[binary]'")
    try:
        import requests
    except ImportError:
        sys.exit("pip install requests")

    params = {"label_event": LABEL_EVENT, "report_event": REPORT_EVENT,
              "bot": args.bot_did, "since": args.since, "limit": args.limit}

    wanted = [k.strip() for k in args.kinds.split(",") if k.strip()]
    unknown = set(wanted) - set(QUERIES)
    if unknown:
        sys.exit(f"unknown kinds: {sorted(unknown)}")

    # cid -> what we know about it, merged across signal kinds
    found: dict[str, dict] = {}
    counts: Counter = Counter()

    with psycopg.connect(args.dsn) as conn:
        conn.read_only = True   # this tool must never write to the moderation DB
        for kind in wanted:
            with conn.cursor() as cur:
                cur.execute(QUERIES[kind], params)
                cols = [d.name for d in cur.description]
                for row in cur.fetchall():
                    r = {k: as_text(v) for k, v in zip(cols, row)}
                    cids = blob_cids(r.get("subjectBlobCids"))
                    if not cids:
                        counts[f"{kind}:no-blobs"] += 1
                        continue
                    labels = classes_from(split_vals(r.get("createLabelVals")))
                    # An appeal is a report with an appeal reasonType. It is the
                    # strongest false-positive evidence available, so separate it
                    # from ordinary reports rather than lumping them together.
                    effective = kind
                    if kind == B_REPORT and report_type(r.get("meta")) in APPEAL_REASONS:
                        effective = B_APPEAL
                    for cid in cids:
                        entry = found.setdefault(cid, {
                            "did": r["subjectDid"], "uri": r["subjectUri"],
                            "kinds": set(), "labels": set()})
                        entry["kinds"].add(effective)
                        if kind in (B_MODEL, B_HUMAN):
                            entry["labels"].update(labels)
                    counts[effective] += 1

    print(f"events matched: " + "  ".join(
        f"{k}={counts[k]:,}" for k in wanted) + f"\ndistinct images: {len(found):,}")
    skipped = {k: v for k, v in counts.items() if k.endswith(":no-blobs")}
    if skipped:
        print("no subjectBlobCids (pre-Feb-2024 or account-level): "
              + "  ".join(f"{k.split(':')[0]}={v:,}" for k, v in skipped.items()))

    images_root = Path(args.images)
    conn_db = connect(args.db)
    existing = {r[0] for r in conn_db.execute("SELECT cid FROM images")}
    now = datetime.now(timezone.utc).isoformat()

    session = requests.Session() if args.apply else None
    written = fetch_failed = already = 0
    by_bucket: Counter = Counter()

    for cid, e in found.items():
        # Most specific signal wins the bucket: a false positive is more useful
        # to a reviewer than the fact that the model also labelled it.
        for bucket in (B_FP, B_APPEAL, B_REPORT, B_HUMAN, B_MODEL):
            if bucket in e["kinds"]:
                break
        by_bucket[bucket] += 1
        if cid in existing:
            already += 1
            continue
        if not args.apply:
            written += 1
            continue

        raw = fetch_image(session, e["did"], cid)
        if raw is None:
            fetch_failed += 1
            continue
        dest = image_path(images_root, cid)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        conn_db.execute(
            "INSERT OR IGNORE INTO images (cid, did, rkey, post_uri, path, bytes,"
            " harvested_at, bucket) VALUES (?,?,?,?,?,?,?,?)",
            (cid, e["did"], "", e["uri"], str(dest.relative_to(images_root)),
             len(raw), now, bucket))

        # A negation says the label was wrong, not what is right, so a false
        # positive carries no label at all -- a human decides what it actually is.
        if bucket != B_FP:
            for label in sorted(e["labels"]):
                source = "ozone-human" if B_HUMAN in e["kinds"] else "ozone-model"
                conn_db.execute(
                    "INSERT OR REPLACE INTO labels (cid,label,source,created_at)"
                    " VALUES (?,?,?,?)", (cid, label, source, now))
        state = NEEDS_DETAIL if bucket in (B_FP, B_REPORT, B_APPEAL) else IMPORTED
        conn_db.execute(
            "INSERT INTO review_state (cid,state,updated_at) VALUES (?,?,?) "
            "ON CONFLICT(cid) DO UPDATE SET state=excluded.state", (cid, state, now))
        written += 1

    if args.apply:
        conn_db.commit()
    conn_db.close()

    print(f"\n{'imported' if args.apply else 'would import'} {written:,} images"
          f"   already present {already:,}"
          + (f"   image fetch failed {fetch_failed:,}" if fetch_failed else ""))
    print("\nby bucket")
    for b, n in by_bucket.most_common():
        print(f"  {b:<22}{n:>7,}")
    print("\nNothing here is written as source='human'. Model labels are the "
          "model's own\npredictions -- training on them entrenches the current "
          "boundary rather than\ncorrecting it, so they arrive as candidates for "
          "review.")
    if not args.apply:
        print("\ndry run -- re-run with --apply to write")


if __name__ == "__main__":
    main()
