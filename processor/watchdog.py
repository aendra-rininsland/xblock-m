#!/usr/bin/env python
"""
Watchdog for xblock-worker and the remote firehose consumer.

Worker checks (every 60s):
- If the worker process is stopped, restart it via supervisorctl.
- If no jobs have been processed for 5 min AND the Redis queue is non-empty
  (jobs waiting but not consumed), restart the worker.
- After MAX_RESTARTS consecutive failed restarts, fire a Windows desktop alert.

Firehose checks (after each worker check):
- Read xblock:firehose:last_enqueue_ts from Redis (written by lib/firehose.ts).
- If the key is stale for >30 min, fire a Windows alert to check pm2 on the
  remote machine. Resets automatically when the firehose recovers.
"""
import logging
import os
import sqlite3
import subprocess
import time
import urllib.request

from dotenv import load_dotenv
load_dotenv()

try:
    import redis as redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s watchdog %(message)s")
logger = logging.getLogger("xblock.watchdog")

BASE            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_DB      = os.path.join(BASE, "logs", "metrics.db")
SUPERVISOR_CONF = os.path.join(BASE, "supervisord.conf")
SUPERVISORCTL   = os.path.expanduser("~/.pyenv/versions/xblock/bin/supervisorctl")

# Set NTFY_TOPIC in processor/.env to enable phone push notifications.
# Set HEALTHCHECKS_URL to a healthchecks.io ping URL for dead-watchdog detection.
NTFY_TOPIC        = os.getenv("NTFY_TOPIC", "")
HEALTHCHECKS_URL  = os.getenv("HEALTHCHECKS_URL", "")

CHECK_INTERVAL           = 60    # seconds between health checks
IDLE_THRESHOLD           = 300   # seconds without a processed job before restart
RESTART_WAIT             = 90    # seconds to wait after restart before rechecking
MAX_RESTARTS             = 3     # consecutive failed restarts before alerting
STARTUP_GRACE            = 120   # grace period after a (re)start before idle check
FIREHOSE_STALE_THRESHOLD = 1800  # 30 min without a firehose enqueue → alert


def sctl(*args) -> tuple[int, str]:
    r = subprocess.run(
        [SUPERVISORCTL, "-c", SUPERVISOR_CONF, *args],
        capture_output=True, text=True, timeout=15,
    )
    return r.returncode, (r.stdout + r.stderr).strip()


def worker_is_running() -> bool:
    _, out = sctl("status", "xblock-worker")
    return "RUNNING" in out


def restart_worker() -> None:
    logger.info("Restarting xblock-worker...")
    code, out = sctl("restart", "xblock-worker")
    logger.info("supervisorctl restart → %s", out)


def last_job_ts() -> float | None:
    """Most recent completed-job timestamp from the local metrics DB."""
    if not os.path.exists(METRICS_DB):
        return None
    try:
        conn = sqlite3.connect(f"file:{METRICS_DB}?mode=ro", uri=True)
        row = conn.execute("SELECT MAX(ts) FROM jobs").fetchone()
        conn.close()
        return float(row[0]) if row and row[0] is not None else None
    except Exception as e:
        logger.warning("Could not read metrics db: %s", e)
        return None


def redis_status() -> tuple[int | None, float | None]:
    """Return (queue_waiting_count, firehose_last_enqueue_ts) in one connection."""
    if not _REDIS_AVAILABLE:
        return None, None
    conn_str = os.getenv("REDIS_CONNECTION_STRING", "")
    if not conn_str:
        return None, None
    try:
        r = redis_lib.from_url(conn_str, socket_timeout=5)
        waiting = int(r.llen("bull:xblock:wait"))
        val = r.get("xblock:firehose:last_enqueue_ts")
        r.close()
        firehose_ts = float(val) if val else None
        return waiting, firehose_ts
    except Exception as e:
        logger.warning("Could not check Redis: %s", e)
        return None, None


def ntfy(title: str, body: str, priority: str = "default", tags: str = "") -> None:
    """Push a notification to ntfy.sh (received on phone via the ntfy app)."""
    if not NTFY_TOPIC:
        return
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=body.encode(),
            headers={"Title": title, "Priority": priority, "Tags": tags},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
        logger.info("ntfy sent: %s", title)
    except Exception as e:
        logger.warning("ntfy failed: %s", e)


def healthchecks_ping(fail: bool = False) -> None:
    """Ping healthchecks.io to signal the watchdog is alive.
    If pings stop arriving (e.g. supervisord died), healthchecks.io alerts via email."""
    if not HEALTHCHECKS_URL:
        return
    url = f"{HEALTHCHECKS_URL.rstrip('/')}/fail" if fail else HEALTHCHECKS_URL
    try:
        urllib.request.urlopen(url, timeout=5)
    except Exception as e:
        logger.warning("healthchecks ping failed: %s", e)


def alert(title: str, body: str, priority: str = "high", tags: str = "warning") -> None:
    """Fire both a Windows balloon and a phone push notification."""
    windows_alert(title, body)
    ntfy(title, body, priority=priority, tags=tags)


def windows_alert(title: str, body: str) -> None:
    """Fire a Windows balloon notification via powershell.exe (available in WSL2)."""
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$n = New-Object System.Windows.Forms.NotifyIcon; "
        "$n.Icon = [System.Drawing.SystemIcons]::Error; "
        "$n.Visible = $true; "
        f"$n.ShowBalloonTip(30000, '{title}', '{body}', [System.Windows.Forms.ToolTipIcon]::Error); "
        "Start-Sleep -Seconds 30; "
        "$n.Dispose()"
    )
    try:
        subprocess.Popen(
            ["powershell.exe", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", script],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        logger.info("Windows alert sent: %s — %s", title, body)
    except Exception as e:
        logger.error("Failed to send Windows alert: %s", e)


def main() -> None:
    logger.info(
        "Started. check_interval=%ds idle_threshold=%ds max_restarts=%d firehose_stale=%ds",
        CHECK_INTERVAL, IDLE_THRESHOLD, MAX_RESTARTS, FIREHOSE_STALE_THRESHOLD,
    )
    ntfy("XBlock Watchdog Started", "Watchdog is running.", priority="low", tags="white_check_mark")

    consecutive_restarts = 0
    worker_alerted       = False
    firehose_alerted     = False
    worker_started_at    = time.time()

    while True:
        time.sleep(CHECK_INTERVAL)
        now = time.time()
        waiting, firehose_ts = redis_status()

        # ── Worker health ─────────────────────────────────────────────────────

        if not worker_is_running():
            logger.warning("Worker is not running, attempting restart...")
            restart_worker()
            consecutive_restarts += 1
            worker_started_at = now
            if consecutive_restarts >= MAX_RESTARTS and not worker_alerted:
                msg = (
                    f"Worker has not recovered after {consecutive_restarts} restart attempts. "
                    "Manual intervention required."
                )
                logger.error(msg)
                alert("XBlock Worker Alert", msg, tags="rotating_light")
                worker_alerted = True
            time.sleep(RESTART_WAIT)
            continue

        ts = last_job_ts()

        # Don't flag idleness during the startup grace window.
        if ts is None and now - worker_started_at < STARTUP_GRACE:
            logger.info("Worker just started, waiting for first job...")
            # Still check firehose below.
        else:
            idle = (now - ts) if ts is not None else float("inf")

            if idle < IDLE_THRESHOLD:
                # Worker is actively processing jobs.
                if consecutive_restarts > 0:
                    logger.info("Worker recovered after %d restart(s).", consecutive_restarts)
                consecutive_restarts = 0
                worker_alerted = False
            else:
                # Worker has been idle — only restart if there are jobs to consume.
                if waiting is not None and waiting == 0:
                    pass  # Empty queue: worker is healthy, firehose may be the issue.
                else:
                    logger.warning(
                        "Worker idle %.0fs with %s waiting job(s) — restarting.",
                        idle, waiting if waiting is not None else "unknown",
                    )
                    restart_worker()
                    consecutive_restarts += 1
                    worker_started_at = now
                    if consecutive_restarts >= MAX_RESTARTS and not worker_alerted:
                        msg = (
                            f"Worker has not processed any jobs after {consecutive_restarts} "
                            "restart attempts. Manual intervention required."
                        )
                        logger.error(msg)
                        alert("XBlock Worker Alert", msg, tags="rotating_light")
                        worker_alerted = True
                    time.sleep(RESTART_WAIT)
                    continue

        # ── Firehose health ───────────────────────────────────────────────────
        # xblock:firehose:last_enqueue_ts is written by lib/firehose.ts each time
        # a batch is successfully pushed to the BullMQ queue. If the key is absent
        # the firehose has never run; if stale, it has silently stopped.

        if firehose_ts is not None:
            firehose_idle = now - firehose_ts
            if firehose_idle > FIREHOSE_STALE_THRESHOLD and not firehose_alerted:
                msg = (
                    f"Firehose may be down: no batches enqueued for {firehose_idle / 60:.0f} min. "
                    "Check pm2 on the remote machine."
                )
                logger.error(msg)
                alert("XBlock Firehose Alert", msg, tags="rotating_light")
                firehose_alerted = True
            elif firehose_idle <= FIREHOSE_STALE_THRESHOLD and firehose_alerted:
                logger.info("Firehose recovered (last enqueue %.0fs ago).", firehose_idle)
                firehose_alerted = False

        healthchecks_ping()


if __name__ == "__main__":
    main()
