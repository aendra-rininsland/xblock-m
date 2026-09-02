#!/usr/bin/env bash
# Application setup for the queue box. Run over SSH after cloud-init finishes:
#
#   scp -i ~/.ssh/xblock-hetzner infra/setup-queue.sh xblock@<ip>:
#   ssh -i ~/.ssh/xblock-hetzner xblock@<ip> 'sudo bash setup-queue.sh'
#
# Idempotent: safe to re-run while iterating.
set -euo pipefail

# Canonical name: xblock-docker 301s here, and git cannot follow that redirect
# on clone -- it prompts for a username, which looks like a private repo.
REPO="${REPO:-https://github.com/aendra-rininsland/xblock-m.git}"
APP_USER=xblock
APP_HOME=/home/$APP_USER
APP_DIR=$APP_HOME/xblock-m
WG_NET=10.10.0
NODE_MAJOR=22

log() { echo "== $*"; }

# ── node ───────────────────────────────────────────────────────────────────
if ! command -v node >/dev/null 2>&1; then
    log "installing node $NODE_MAJOR (arm64)"
    curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash -
    apt-get install -y nodejs
fi
log "node $(node --version)"

command -v pm2 >/dev/null 2>&1 || { log "installing pm2"; npm install -g pm2; }

# ── wireguard ──────────────────────────────────────────────────────────────
# Love sits behind Windows NAT and cannot accept inbound connections, so it is
# the client and this box is the server. That is also why plain WireGuard is
# enough here and Tailscale's NAT traversal is not needed.
if [ ! -f /etc/wireguard/server-private.key ]; then
    log "generating WireGuard server keys"
    umask 077
    mkdir -p /etc/wireguard
    wg genkey | tee /etc/wireguard/server-private.key | wg pubkey > /etc/wireguard/server-public.key
    chmod 600 /etc/wireguard/server-private.key
fi

if [ ! -f /etc/wireguard/wg0.conf ]; then
    log "writing wg0.conf"
    cat > /etc/wireguard/wg0.conf <<EOF
[Interface]
Address    = ${WG_NET}.1/24
ListenPort = 51820
PrivateKey = $(cat /etc/wireguard/server-private.key)

# Peers are appended by add-wg-peer.sh -- do not hand-edit below this line.
EOF
    chmod 600 /etc/wireguard/wg0.conf
fi

systemctl enable --now wg-quick@wg0 2>/dev/null || systemctl restart wg-quick@wg0

# ── redis auth ─────────────────────────────────────────────────────────────
# Defence in depth. The tunnel already means Redis is unreachable from the
# internet; the password covers anything that reaches the tunnel.
if [ ! -f /etc/redis/redis.conf.d/auth.conf ]; then
    log "generating redis password"
    PASS=$(head -c 32 /dev/urandom | base64 | tr -d '/+=' | head -c 40)
    printf 'requirepass %s\n' "$PASS" > /etc/redis/redis.conf.d/auth.conf
    chown redis:redis /etc/redis/redis.conf.d/auth.conf
    chmod 640 /etc/redis/redis.conf.d/auth.conf
    # Written for the operator to copy to Love's .env, then delete.
    printf '%s\n' "$PASS" > /root/redis-password.txt
    chmod 600 /root/redis-password.txt
    log "redis password written to /root/redis-password.txt -- copy it, then delete it"
fi
systemctl restart redis-server

# ── application ────────────────────────────────────────────────────────────
if [ ! -d "$APP_DIR" ]; then
    log "cloning $REPO"
    sudo -u $APP_USER git clone "$REPO" "$APP_DIR"
fi

cd "$APP_DIR"
sudo -u $APP_USER git pull --ff-only || true
sudo -u $APP_USER npm ci --omit=dev 2>/dev/null || sudo -u $APP_USER npm install --omit=dev

if [ ! -f "$APP_DIR/.env" ]; then
    log "writing .env"
    sudo -u $APP_USER tee "$APP_DIR/.env" >/dev/null <<EOF
REDIS_HOSTNAME=127.0.0.1
EOF
fi

log "starting under pm2"
sudo -u $APP_USER bash -c "cd $APP_DIR && pm2 start ecosystem.config.js || pm2 reload ecosystem.config.js"
sudo -u $APP_USER pm2 save
env PATH="$PATH:/usr/bin" pm2 startup systemd -u $APP_USER --hp "$APP_HOME" >/dev/null

log "done. Next: add Love as a WireGuard peer with add-wg-peer.sh"
wg show
