#!/usr/bin/env bash
# Ozone host provision. Run on a fresh CAX21 (Ubuntu 24.04, arm64) as root.
#
#   ssh -i ~/.ssh/xblock-hetzner root@<ip> 'bash -s' < infra/setup-ozone.sh
#
# Brings the box to the point where it can accept the database, but deliberately
# does NOT start Ozone or restore. Those are cutover steps -- see RESTORE below
# and infra/README.md -- because starting Ozone against an empty database would
# let it run its own migrations and create a schema the dump then conflicts with.
set -euo pipefail

log() { echo "== $*"; }

log "packages"
apt-get update
apt-get install -y ca-certificates curl gnupg ufw postgresql-client-14 rsync jq

log "docker"
if ! command -v docker >/dev/null 2>&1; then
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=arm64 signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
        > /etc/apt/sources.list.d/docker.list
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
fi
systemctl enable --now docker

log "firewall"
# Ozone is a public service: 80 and 443 must be open. Postgres must NOT be --
# on the old box it listens on 0.0.0.0:5432, which is only survivable because a
# security group sits in front of it. Here there is no security group, so the
# host firewall is the control.
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow OpenSSH
ufw allow 80/tcp  comment 'HTTP (ACME + redirect)'
ufw allow 443/tcp comment 'HTTPS + label firehose'
ufw --force enable

log "directories"
mkdir -p /ozone/postgres /ozone/caddy/data /ozone/caddy/etc/caddy /opt/xblock /var/backups/ozone
chown -R 999:999 /ozone/postgres   # postgres uid in the official image

log "backup cron"
install -m 0755 /dev/stdin /opt/xblock/pg-backup.sh <<'PLACEHOLDER'
#!/bin/sh
echo "copy infra/pg-backup.sh here before enabling this cron entry" >&2
exit 1
PLACEHOLDER
cat > /etc/cron.d/xblock-backup <<'CRON'
# Nightly Ozone backup. Verified before upload; see infra/pg-backup.sh.
15 4 * * * root /opt/xblock/pg-backup.sh >> /var/log/xblock-backup.log 2>&1
CRON

cat <<'NEXT'

== base ready ==

Remaining, in order — these are cutover steps, not provisioning:

  1. Copy the stack config across VERBATIM from the old host. These hold the
     signing key and admin credentials; do not retype them.
       rsync -av --rsync-path='sudo rsync' \
         ubuntu@xblock.aendra.dev:/ozone/caddy/ /ozone/caddy/
       for f in ozone.env ozone_news.env postgres.env; do
         ssh ubuntu@xblock.aendra.dev "sudo cat /ozone/$f" > /ozone/$f
         chmod 600 /ozone/$f
       done

  2. Copy infra/ozone-compose.yaml to /ozone/compose.yaml and
     infra/pg-backup.sh to /opt/xblock/pg-backup.sh.

  3. Start ONLY postgres, then restore. Ozone must not touch an empty database.
       docker compose -f /ozone/compose.yaml up -d postgres
       # stream the dump in from wherever it lives
       pg_restore -h 127.0.0.1 -U postgres -d ozone --no-owner --no-acl -j 2 ozone.dump

  4. VERIFY before starting Ozone:
       signing_key row count matches the old host
       label min(id)/max(id)/count match  <- subscribeLabels cursors depend on it
       moderation_event count matches

  5. docker compose -f /ozone/compose.yaml up -d

  6. Only then flip DNS. TTL must already be lowered (currently 14400).

NEXT
