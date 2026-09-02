#!/usr/bin/env bash
# One-shot bootstrap for a BARE Ubuntu 24.04 box (arm64 or x86_64), as root.
#
# Does everything cloud-init-queue.yaml + setup-queue.sh do combined, so the
# server can be created by hand in the Hetzner console with no user-data pasted
# in. Use this when the box was made manually; use the cloud-init file when it
# was made through the API.
#
#   ssh -i ~/.ssh/xblock-hetzner root@<ip> 'bash -s' < infra/bootstrap-queue.sh
#
# Idempotent -- safe to re-run.
set -uo pipefail

APP_USER=xblock
APP_HOME=/home/$APP_USER
APP_DIR=$APP_HOME/xblock-m
# Canonical name (xblock-docker 301s here). The repo is PUBLIC.
REPO="${REPO:-https://github.com/aendra-rininsland/xblock-m.git}"
BRANCH="${BRANCH:-feat/infra-migration}"
WG_NET=10.10.0
NODE_MAJOR=22
PUBKEY='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJ71N05nNen+OjDWUEPwD3n6NErdeBOkmKGk6fx7VJT2 xblock-migration-2026-09'

log() { echo "== $*"; }
log "arch: $(uname -m)  (expect aarch64)"

# ── user ───────────────────────────────────────────────────────────────────
if ! id "$APP_USER" >/dev/null 2>&1; then
    log "creating $APP_USER"
    adduser --disabled-password --gecos "" "$APP_USER"
    usermod -aG sudo "$APP_USER"
    echo "$APP_USER ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$APP_USER
    chmod 440 /etc/sudoers.d/$APP_USER
fi
install -d -m 700 -o "$APP_USER" -g "$APP_USER" "$APP_HOME/.ssh"
grep -qF "$PUBKEY" "$APP_HOME/.ssh/authorized_keys" 2>/dev/null || \
    echo "$PUBKEY" >> "$APP_HOME/.ssh/authorized_keys"
chown "$APP_USER:$APP_USER" "$APP_HOME/.ssh/authorized_keys"
chmod 600 "$APP_HOME/.ssh/authorized_keys"

# ── packages ───────────────────────────────────────────────────────────────
log "packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq redis-server wireguard ufw git curl jq fail2ban \
    unattended-upgrades ca-certificates

if ! command -v node >/dev/null 2>&1; then
    log "node $NODE_MAJOR"
    curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash - >/dev/null
    apt-get install -y -qq nodejs
fi
log "node $(node --version)"
command -v pm2 >/dev/null 2>&1 || npm install -g pm2 >/dev/null 2>&1

# ── firewall ───────────────────────────────────────────────────────────────
# Redis is NOT opened. It is reached through the tunnel only; the old box's
# public 6379 behind a security-group allowlist is exactly what this replaces.
log "firewall"
ufw --force reset >/dev/null
ufw default deny incoming >/dev/null
ufw default allow outgoing >/dev/null
ufw allow OpenSSH >/dev/null
ufw allow 51820/udp comment 'WireGuard' >/dev/null
ufw --force enable >/dev/null

# ── redis ──────────────────────────────────────────────────────────────────
log "redis config"
mkdir -p /etc/redis/redis.conf.d
cat > /etc/redis/redis.conf.d/xblock.conf <<EOF
# The "-" prefix makes the bind OPTIONAL. The WireGuard address only exists
# once wg-quick@wg0 is up, so without it redis loses the boot race and exits 1.
# systemd retries and it comes up second time, which hides the problem while
# leaving every reboot a coin flip.
bind 127.0.0.1 -${WG_NET}.1
protected-mode yes
port 6379

# 4 GB box; observed peak on the old instance was 2.05 GB. This is an OOM
# backstop, not a tuning knob. See the note in processor/constants.py: a memory
# cap acts as a depth cap, which discards the NEWEST work during a burst -- the
# opposite of what the LIFO design wants. The 48h age bound is the intended
# limiter. If this is ever actually hit, move to a CAX21 rather than lowering it.
maxmemory 2560mb
maxmemory-policy noeviction

# The queue is reconstructible: jobs expire after 48h and the firehose replays.
save ""
appendonly no
EOF
# Redis `include` takes ONE path and does not expand globs, so every file in
# conf.d needs its own line. Including only xblock.conf silently leaves
# auth.conf unread -- requirepass is generated, written, and never applied, and
# `redis-cli ping` answers PONG to anyone who reaches the port. Verify with
# `redis-cli ping` expecting NOAUTH, not by reading the config.
grep -q "redis.conf.d/xblock.conf" /etc/redis/redis.conf || \
    echo "include /etc/redis/redis.conf.d/xblock.conf" >> /etc/redis/redis.conf
grep -q "redis.conf.d/auth.conf" /etc/redis/redis.conf || \
    echo "include /etc/redis/redis.conf.d/auth.conf" >> /etc/redis/redis.conf

if [ ! -f /etc/redis/redis.conf.d/auth.conf ]; then
    PASS=$(head -c 32 /dev/urandom | base64 | tr -d '/+=' | head -c 40)
    printf 'requirepass %s\n' "$PASS" > /etc/redis/redis.conf.d/auth.conf
    printf '%s\n' "$PASS" > /root/redis-password.txt
    chmod 600 /root/redis-password.txt
fi
chown -R redis:redis /etc/redis/redis.conf.d
systemctl enable --now redis-server >/dev/null 2>&1
systemctl restart redis-server

# ── wireguard server side (unused by default; the tunnel is SSH) ────────────
if [ ! -f /etc/wireguard/server-private.key ]; then
    log "wireguard keys"
    umask 077; mkdir -p /etc/wireguard
    wg genkey | tee /etc/wireguard/server-private.key | wg pubkey > /etc/wireguard/server-public.key
    cat > /etc/wireguard/wg0.conf <<EOF
[Interface]
Address    = ${WG_NET}.1/24
ListenPort = 51820
PrivateKey = $(cat /etc/wireguard/server-private.key)
EOF
    chmod 600 /etc/wireguard/wg0.conf /etc/wireguard/server-private.key
fi
systemctl enable --now wg-quick@wg0 >/dev/null 2>&1 || true

# ── git over HTTP/1.1 ──────────────────────────────────────────────────────
# Ubuntu 24.04 ships git 2.43.0 against libcurl 8.5.0 + nghttp2 1.59.0, and
# that combination mis-frames GitHub's smart-HTTP ref advertisement over HTTP/2.
# git reports:
#     fatal: expected flush after ref listing
#     fatal: could not read Username for 'https://github.com'
# The username prompt is a FALLBACK after the parse failure, not an auth
# problem -- the repo is public and plain curl fetches the same URL fine over
# HTTP/2. Without this pin, provisioning looks like a private-repo issue and
# sends you hunting for deploy keys that were never needed.
git config --system http.version HTTP/1.1

# ── application ────────────────────────────────────────────────────────────
if [ ! -d "$APP_DIR" ]; then
    log "cloning $REPO ($BRANCH)"
    sudo -u "$APP_USER" git clone -b "$BRANCH" "$REPO" "$APP_DIR"
fi
cd "$APP_DIR" || exit 1
sudo -u "$APP_USER" git fetch origin "$BRANCH" -q && sudo -u "$APP_USER" git checkout "$BRANCH" -q
sudo -u "$APP_USER" npm install --omit=dev --silent 2>&1 | tail -2

[ -f "$APP_DIR/.env" ] || sudo -u "$APP_USER" tee "$APP_DIR/.env" >/dev/null <<EOF
REDIS_HOSTNAME=127.0.0.1
EOF

systemctl enable --now fail2ban >/dev/null 2>&1

cat <<DONE

== bootstrap complete ==
  node       $(node --version)
  redis      $(systemctl is-active redis-server)
  firewall   $(ufw status | head -1)
  repo       $APP_DIR @ $(cd "$APP_DIR" && git rev-parse --short HEAD)

  Redis password is in /root/redis-password.txt
  Redis is NOT started under pm2 yet -- the firehose is cut over deliberately,
  not automatically, so the old box keeps consuming until we switch.
DONE
