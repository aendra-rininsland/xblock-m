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
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "processor"))

from label_values import model_class  # noqa: E402
from manifest import (IMPORTED, NEEDS_DETAIL, connect, default_db,
                      default_images, image_path)

CDN = "https://cdn.bsky.app/img/feed_thumbnail/plain/{did}/{cid}@jpeg"

# Reports carry no subjectBlobCids -- not sometimes, ever. Every one of the
# 45,993 report events in the store has the column empty, because Ozone only
# denormalises blob CIDs onto the events it creates from a label, not from a
# user's report. The images are still recoverable: the report names the post in
# `subjectUri`, and the post record itself lists its blobs. So resolve the record
# through the AppView and read the CIDs off it.
#
# The public AppView needs no auth and takes 25 URIs per call, which is what
# makes this practical -- ~39k reported posts is ~1.6k requests, not 39k.
APPVIEW = "https://public.api.bsky.app/xrpc/app.bsky.feed.getPosts"
GET_POSTS_MAX = 25          # the lexicon's own cap on `uris`

# Only post records have blobs. Reports also target accounts (a bare DID) and the
# occasional non-post record; neither resolves to an image.
POST_URI = "at://%/app.bsky.feed.post/%"

LABEL_EVENT = "tools.ozone.moderation.defs#modEventLabel"
REPORT_EVENT = "tools.ozone.moderation.defs#modEventReport"
APPEAL_REASONS = ("com.atproto.moderation.defs#reasonAppeal",
                  "tools.ozone.report.defs#reasonAppeal")

# Buckets, so the review UI can work each kind of signal separately.
B_CORRECTED = "ozone-corrected"  # a human removed a label AND supplied the right one
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


def image_cids(embed) -> list[str]:
    """Blob CIDs of the images on a post record.

    Two embed shapes carry them: `app.bsky.embed.images` directly, and
    `app.bsky.embed.recordWithMedia`, which nests the same structure under
    `media`. A plain `app.bsky.embed.record` -- a quote post -- is deliberately
    skipped: those images belong to the post being quoted, and a report is about
    the post that was reported, not the one it points at."""
    if not isinstance(embed, dict):
        return []
    kind = embed.get("$type") or ""
    if kind.startswith("app.bsky.embed.recordWithMedia"):
        return image_cids(embed.get("media"))
    if not kind.startswith("app.bsky.embed.images"):
        return []
    cids = []
    for img in embed.get("images") or []:
        if not isinstance(img, dict):
            continue
        ref = (img.get("image") or {}).get("ref")
        link = ref.get("$link") if isinstance(ref, dict) else ref
        if isinstance(link, str) and link:
            cids.append(link)
    return cids


def resolve_posts(session, uris: list[str], progress=True) -> dict[str, list[str]]:
    """at:// post URIs -> their image blob CIDs, via the public AppView.

    A URI absent from the response is a post that no longer resolves: deleted,
    or in a repo that is no longer served. Over a backlog this old that is
    routine, not an error, so it is counted rather than raised."""
    out: dict[str, list[str]] = {}
    total = (len(uris) + GET_POSTS_MAX - 1) // GET_POSTS_MAX
    for n, i in enumerate(range(0, len(uris), GET_POSTS_MAX), start=1):
        batch = uris[i:i + GET_POSTS_MAX]
        for attempt in range(4):
            try:
                resp = session.get(APPVIEW, params={"uris": batch}, timeout=30)
            except Exception:
                time.sleep(2 * (attempt + 1))
                continue
            # The AppView is a shared public service; back off when it says to
            # rather than hammering it through a 39k-post backlog.
            if resp.status_code == 429:
                wait = resp.headers.get("Retry-After")
                time.sleep(int(wait) if (wait or "").isdigit() else 5 * (attempt + 1))
                continue
            if resp.status_code != 200:
                break
            try:
                posts = resp.json().get("posts") or []
            except ValueError:
                break
            for p in posts:
                uri, rec = p.get("uri"), p.get("record")
                if isinstance(uri, str) and isinstance(rec, dict):
                    cids = image_cids(rec.get("embed"))
                    if cids:
                        out[uri] = cids
            break
        if progress and (n % 20 == 0 or n == total):
            print(f"  resolving posts via AppView: {n}/{total} batches, "
                  f"{len(out):,} with images", end="\r", flush=True)
    if progress and total:
        print()
    return out


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
    p.add_argument("--resolve-blobs", action="store_true",
                   help="for events with no subjectBlobCids, resolve the post "
                        "record through the AppView and read its blob CIDs. "
                        "This is the only way to get images out of reports, "
                        "which never carry blob CIDs of their own.")
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
    # post uri -> what we know, for events whose blob CIDs have to be resolved
    deferred: dict[str, dict] = {}
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
                    labels = classes_from(split_vals(r.get("createLabelVals")))
                    # An appeal is a report with an appeal reasonType. It is the
                    # strongest false-positive evidence available, so separate it
                    # from ordinary reports rather than lumping them together.
                    effective = kind
                    if kind == B_REPORT and report_type(r.get("meta")) in APPEAL_REASONS:
                        effective = B_APPEAL
                    if not cids:
                        # No blob CIDs on the event. If the subject is a post
                        # record the images are still reachable through the
                        # AppView, so hold the URI back for a second pass rather
                        # than dropping the event on the floor.
                        uri = r.get("subjectUri") or ""
                        if args.resolve_blobs and uri.startswith("at://") \
                                and "/app.bsky.feed.post/" in uri:
                            e = deferred.setdefault(uri, {"kinds": set(), "labels": set()})
                            e["kinds"].add(effective)
                            if kind in (B_MODEL, B_HUMAN):
                                e["labels"].update(labels)
                            counts[f"{effective}:deferred"] += 1
                        else:
                            counts[f"{kind}:no-blobs"] += 1
                        continue
                    for cid in cids:
                        entry = found.setdefault(cid, {
                            "did": r["subjectDid"], "uri": r["subjectUri"],
                            "kinds": set(), "labels": set()})
                        entry["kinds"].add(effective)
                        if kind in (B_MODEL, B_HUMAN):
                            entry["labels"].update(labels)
                    counts[effective] += 1

    # Second pass: turn the deferred post URIs into blob CIDs. This is what makes
    # reports usable at all -- without it the report bucket yields exactly zero
    # images, however high --limit is set.
    if deferred:
        import requests as _requests
        print(f"resolving {len(deferred):,} post records through the AppView "
              f"(no blob CIDs on the event)...")
        resolver = _requests.Session()
        resolved = resolve_posts(resolver, list(deferred))
        resolver.close()
        gained = 0
        for uri, e in deferred.items():
            cids = resolved.get(uri)
            if not cids:
                # Either the post is gone, or it never had images -- a report on
                # a text-only post is still a real report, just not training data.
                counts["resolve:no-images"] += 1
                continue
            did = uri.split("/")[2] if uri.count("/") >= 2 else ""
            for cid in cids:
                entry = found.setdefault(cid, {"did": did, "uri": uri,
                                               "kinds": set(), "labels": set()})
                entry["kinds"].update(e["kinds"])
                entry["labels"].update(e["labels"])
                gained += 1
        for k in list(counts):
            if k.endswith(":deferred"):
                counts[k.removesuffix(":deferred")] += counts[k]
        print(f"  resolved {len(resolved):,} posts -> {gained:,} image references")

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
        # A negation and a human's label are two different pieces of evidence
        # about the same image, and only the negation is uninformative. When a
        # moderator both removed a label and applied one -- in a single relabel
        # event, or as a later correction on the same subject -- the second half
        # is ground truth: not "X was wrong" but "it is actually Y".
        #
        # Without this, such an image falls into B_FP, whose whole point is that
        # it carries no label, and the moderator's answer is thrown away. It then
        # goes back to review so a human can supply the answer a human already
        # gave. That is the single largest avoidable loss in this importer.
        if B_FP in e["kinds"] and B_HUMAN in e["kinds"] and e["labels"]:
            e["kinds"].add(B_CORRECTED)

        # Most specific signal wins the bucket: a false positive is more useful
        # to a reviewer than the fact that the model also labelled it.
        for bucket in (B_CORRECTED, B_FP, B_APPEAL, B_REPORT, B_HUMAN, B_MODEL):
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

        # A bare negation says the label was wrong, not what is right, so a false
        # positive carries no label at all -- a human decides what it actually is.
        # B_CORRECTED is the exception, and deliberately not B_FP: there a human
        # supplied the replacement, so the label written here is that human's.
        if bucket != B_FP:
            for label in sorted(e["labels"]):
                source = "ozone-human" if B_HUMAN in e["kinds"] else "ozone-model"
                conn_db.execute(
                    "INSERT OR REPLACE INTO labels (cid,label,source,created_at)"
                    " VALUES (?,?,?,?)", (cid, label, source, now))
        # Corrected images arrive WITH their label but still go to review. The
        # corpus's guarantee is that a human confirmed every label in it, and a
        # moderator's action in Ozone is not the same act as labelling for
        # training -- so it is a confirm pass, not a free pass.
        state = (NEEDS_DETAIL
                 if bucket in (B_CORRECTED, B_FP, B_REPORT, B_APPEAL)
                 else IMPORTED)
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
