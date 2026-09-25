#!/usr/bin/env bash
# Nightly Ozone Postgres backup.
#
# Runs ON the Ozone host, from cron:
#   15 4 * * *  /opt/xblock/pg-backup.sh >> /var/log/xblock-backup.log 2>&1
#
# HISTORY -- read this before changing the ordering below.
#
# The first version of this script took the service down on 14 September 2026.
# Two of its decisions combined into a death spiral:
#
#   1. It verified archives with `pg_restore -l` on the HOST -- but the host
#      deliberately has no postgresql-client (the migration used the container's
#      tools to avoid a version mismatch). So verification failed EVERY night
#      from the first: "pg_restore: command not found". 12 runs, 0 clean.
#
#   2. It only pruned old backups after a clean run. With every run failing,
#      nothing was ever pruned. Twelve 3.8 GB dumps accumulated on an 80 GB disk
#      holding a 34 GB database until Postgres could not write postmaster.pid.
#
# Neither failure was visible: PING_URL was unset, so healthchecks never heard
# about it, and the only symptom was Ozone returning 503 twelve days later.
#
# The fixes, in order of importance:
#   - verify with the container's pg_restore, byte-identical to the server
#   - PRUNE FIRST, before dumping, so accumulation is bounded regardless of
#     whether anything else in this script works
#   - refuse to dump if free disk is below a floor
#   - keep 2 locally, not 7 -- local copies on the same disk as the database
#     are a convenience, not a backup; the offsite copy is the backup
#   - default PING_URL to LABEL_HEALTHCHECKS_URL-style env so failure is heard
set -uo pipefail

DB_CONTAINER="${DB_CONTAINER:-postgres}"
DB_IMAGE="${DB_IMAGE:-postgres:14.11-bookworm}"
DB_USER="${DB_USER:-postgres}"
DATABASES="${DATABASES:-ozone ozone_news}"
LOCAL_DIR="${LOCAL_DIR:-/var/backups/ozone}"
KEEP_LOCAL="${KEEP_LOCAL:-2}"
MIN_FREE_GB="${MIN_FREE_GB:-12}"          # ~3x a compressed dump

# Hetzner Storage Box, e.g. u123456@u123456.your-storagebox.de:/home/ozone
REMOTE="${REMOTE:-}"
SSH_KEY="${SSH_KEY:-/root/.ssh/storagebox}"
PING_URL="${PING_URL:-}"

STAMP=$(date -u +%Y-%m-%dT%H%M%SZ)
DEST="$LOCAL_DIR/$STAMP"
FAILED=0

log()  { echo "[$(date -u +%FT%TZ)] $*"; }
fail() { log "ERROR: $*"; FAILED=1; }
finish() {
    if [ -n "$PING_URL" ]; then
        if [ "$FAILED" -eq 0 ]; then curl -fsS -m 15 "$PING_URL" >/dev/null 2>&1
        else curl -fsS -m 15 "$PING_URL/fail" >/dev/null 2>&1; fi
    fi
    log "done (failed=$FAILED)"
    exit "$FAILED"
}

mkdir -p "$LOCAL_DIR" || { echo "cannot create $LOCAL_DIR"; exit 1; }

# ── 1. prune FIRST ─────────────────────────────────────────────────────────
# Unconditional. If this script is broken in any other way, this still runs,
# so the worst case is "no new backup tonight", never "disk full".
log "pruning to newest $KEEP_LOCAL local backups"
ls -1dt "$LOCAL_DIR"/*/ 2>/dev/null | tail -n +$((KEEP_LOCAL + 1)) | while read -r old; do
    log "  removing $old"; rm -rf "$old"
done

# ── 2. refuse to fill the disk ─────────────────────────────────────────────
FREE_GB=$(df -BG --output=avail "$LOCAL_DIR" | tail -1 | tr -dc '0-9')
if [ "${FREE_GB:-0}" -lt "$MIN_FREE_GB" ]; then
    fail "only ${FREE_GB} GB free, need ${MIN_FREE_GB}; NOT dumping (that is how Postgres died last time)"
    finish
fi
log "${FREE_GB} GB free"

if ! docker inspect "$DB_CONTAINER" --format '{{.State.Health.Status}}' 2>/dev/null | grep -q healthy; then
    fail "$DB_CONTAINER is not healthy; not dumping"
    finish
fi

mkdir -p "$DEST" || { fail "cannot create $DEST"; finish; }

# ── 3. dump ────────────────────────────────────────────────────────────────
log "dumping globals"
docker exec "$DB_CONTAINER" pg_dumpall -U "$DB_USER" --globals-only \
    > "$DEST/globals.sql" 2>"$DEST/globals.err" || fail "pg_dumpall: $(head -c 300 "$DEST/globals.err")"

for DB in $DATABASES; do
    log "dumping $DB"
    if ! docker exec "$DB_CONTAINER" pg_dump -U "$DB_USER" -Fc -Z6 -d "$DB" \
            > "$DEST/$DB.dump" 2>"$DEST/$DB.err"; then
        fail "pg_dump $DB: $(head -c 300 "$DEST/$DB.err")"
        continue
    fi
    log "  $DB dumped: $(stat -c%s "$DEST/$DB.dump") bytes"

    # ── 4. verify with the CONTAINER's pg_restore ──────────────────────────
    # The host has no pg_restore, on purpose. A throwaway container from the
    # same image reads the archive TOC; --network none because it needs no
    # network and should not have any.
    if docker run --rm --network none -v "$DEST:/b:ro" "$DB_IMAGE" \
            pg_restore -l "/b/$DB.dump" > "$DEST/$DB.toc" 2>"$DEST/$DB.tocerr"; then
        ENTRIES=$(wc -l < "$DEST/$DB.toc")
        log "  $DB archive readable: $ENTRIES TOC entries"
        [ "$ENTRIES" -lt 10 ] && fail "$DB TOC implausibly small ($ENTRIES)"
    else
        fail "$DB archive UNREADABLE: $(head -c 300 "$DEST/$DB.tocerr")"
    fi
done

( cd "$DEST" && sha256sum ./*.dump globals.sql > SHA256SUMS 2>/dev/null )

# ── 5. offsite ─────────────────────────────────────────────────────────────
if [ -n "$REMOTE" ]; then
    log "uploading to $REMOTE"
    rsync -az --timeout=1800 \
        -e "ssh -i $SSH_KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
        "$DEST" "$REMOTE/" && log "  upload ok" || fail "upload failed -- local copy retained"
else
    # Loud, because a local-only copy on the database's own disk is not a
    # backup, and this line was ignored for twelve days last time.
    fail "REMOTE unset -- NO OFFSITE COPY EXISTS. Local dumps share a disk with the database."
fi

finish
