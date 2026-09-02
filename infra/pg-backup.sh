#!/usr/bin/env bash
# Nightly Ozone Postgres backup.
#
# Before 2 September 2026 this database had never been backed up. It holds the
# labeller's signing key and every label ever published; losing it is not
# recoverable from anywhere else, images included. That makes this the highest
# value script in infra/ and the reason it verifies rather than assuming.
#
# Runs ON the Ozone host, from cron:
#   15 4 * * *  /opt/xblock/pg-backup.sh >> /var/log/xblock-backup.log 2>&1
#
# Dumps locally, VERIFIES the archive is readable, then ships it. Verifying
# before upload matters: an unverified backup is a belief, not a backup, and the
# failure mode is discovering it at restore time.
#
# NOTE on disk: the old t2.large had 23 GB free against a 33 GB database and
# could not hold a dump at all -- there, this must be run from elsewhere and
# streamed. On the CAX21 (80 GB, ~40 GB free, ~4 GB compressed dump) local-then-
# upload is safe and lets us verify first.
set -uo pipefail

DB_CONTAINER="${DB_CONTAINER:-postgres}"
DB_USER="${DB_USER:-postgres}"
DATABASES="${DATABASES:-ozone ozone_news}"
LOCAL_DIR="${LOCAL_DIR:-/var/backups/ozone}"
KEEP_LOCAL="${KEEP_LOCAL:-7}"

# Hetzner Storage Box, e.g. u123456@u123456.your-storagebox.de:/home/ozone
REMOTE="${REMOTE:-}"
SSH_KEY="${SSH_KEY:-/root/.ssh/storagebox}"
PING_URL="${PING_URL:-}"

STAMP=$(date -u +%Y-%m-%dT%H%M%SZ)
DEST="$LOCAL_DIR/$STAMP"
FAILED=0

log() { echo "[$(date -u +%FT%TZ)] $*"; }
fail() { log "ERROR: $*"; FAILED=1; }

mkdir -p "$DEST" || { echo "cannot create $DEST"; exit 1; }

# Roles and grants. Small, and a restore without them silently loses ownership.
log "dumping globals"
if ! docker exec "$DB_CONTAINER" pg_dumpall -U "$DB_USER" --globals-only \
        > "$DEST/globals.sql" 2>"$DEST/globals.err"; then
    fail "pg_dumpall failed: $(head -c 300 "$DEST/globals.err")"
fi

for DB in $DATABASES; do
    log "dumping $DB"
    if docker exec "$DB_CONTAINER" pg_dump -U "$DB_USER" -Fc -Z6 -d "$DB" \
            > "$DEST/$DB.dump" 2>"$DEST/$DB.err"; then
        SIZE=$(stat -c%s "$DEST/$DB.dump")
        log "  $DB dumped: $SIZE bytes"

        # The verification step. pg_restore -l walks the whole archive TOC, so a
        # truncated or corrupt dump fails here rather than during an outage.
        if pg_restore -l "$DEST/$DB.dump" > "$DEST/$DB.toc" 2>"$DEST/$DB.tocerr"; then
            ENTRIES=$(wc -l < "$DEST/$DB.toc")
            log "  $DB archive readable: $ENTRIES TOC entries"
            [ "$ENTRIES" -lt 10 ] && fail "$DB TOC implausibly small ($ENTRIES)"
        else
            fail "$DB archive UNREADABLE: $(head -c 300 "$DEST/$DB.tocerr")"
        fi
    else
        fail "pg_dump $DB failed: $(head -c 300 "$DEST/$DB.err")"
    fi
done

log "checksums"
( cd "$DEST" && sha256sum ./*.dump globals.sql > SHA256SUMS 2>/dev/null )

if [ -n "$REMOTE" ]; then
    log "uploading to $REMOTE"
    if rsync -az --timeout=1800 -e "ssh -i $SSH_KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
            "$DEST" "$REMOTE/"; then
        log "  upload ok"
    else
        fail "upload failed -- local copy retained at $DEST"
    fi
else
    log "REMOTE unset; keeping local only (offsite copy NOT made)"
fi

# Prune only after a successful run, so a broken night never deletes the last
# good backup.
if [ "$FAILED" -eq 0 ]; then
    log "pruning local beyond $KEEP_LOCAL"
    ls -1dt "$LOCAL_DIR"/*/ 2>/dev/null | tail -n +$((KEEP_LOCAL + 1)) | while read -r old; do
        log "  removing $old"; rm -rf "$old"
    done
else
    log "failures occurred; skipping prune"
fi

if [ -n "$PING_URL" ]; then
    [ "$FAILED" -eq 0 ] && curl -fsS -m 15 "$PING_URL" >/dev/null 2>&1 \
                        || curl -fsS -m 15 "$PING_URL/fail" >/dev/null 2>&1
fi

log "done (failed=$FAILED)"
exit "$FAILED"
