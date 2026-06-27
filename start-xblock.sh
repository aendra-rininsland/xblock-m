#!/usr/bin/env bash
# Starts (or reloads) supervisord for the XBlock service.
# Safe to run multiple times — reloads config if already running.
set -euo pipefail

CONF="/home/aendra/xblock-docker/supervisord.conf"
PID_FILE="/tmp/xblock-supervisord.pid"
SUPERVISORD="/home/aendra/.pyenv/versions/xblock/bin/supervisord"
SUPERVISORCTL="/home/aendra/.pyenv/versions/xblock/bin/supervisorctl"

mkdir -p /home/aendra/xblock-docker/logs

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "supervisord already running (pid $(cat "$PID_FILE")), reloading config..."
    "$SUPERVISORCTL" -c "$CONF" reread
    "$SUPERVISORCTL" -c "$CONF" update
else
    echo "Starting supervisord..."
    "$SUPERVISORD" -c "$CONF"
fi
