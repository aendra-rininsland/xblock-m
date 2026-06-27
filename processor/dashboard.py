#!/usr/bin/env python
"""XBlock monitoring dashboard."""
import asyncio
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

app = FastAPI(title="XBlock Dashboard")

LOGS_DIR = Path(__file__).parent.parent / "logs"
METRICS_FILE = LOGS_DIR / "metrics.jsonl"
WORKER_LOG = LOGS_DIR / "worker.log"
SUPERVISOR_CONF = Path(__file__).parent.parent / "supervisord.conf"


def supervisorctl(*args: str) -> str:
    result = subprocess.run(
        ["supervisorctl", "-c", str(SUPERVISOR_CONF), *args],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return (result.stdout + result.stderr).strip()


@app.get("/api/status")
async def get_status():
    raw = supervisorctl("status", "xblock-worker")
    parts = raw.split()
    status = parts[1] if len(parts) >= 2 else "UNKNOWN"
    uptime = " ".join(parts[3:]) if len(parts) > 3 else ""
    return {"status": status, "uptime": uptime, "raw": raw}


@app.post("/api/control/{action}")
async def control(action: str):
    if action not in ("start", "stop", "restart"):
        raise HTTPException(status_code=400, detail="invalid action")
    output = supervisorctl(action, "xblock-worker")
    return {"output": output}


@app.get("/api/metrics")
async def get_metrics():
    now = time.time()
    midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    thirty_min_ago = now - 1800

    jobs_today = 0
    jobs_last_minute = 0
    total_duration = 0.0
    duration_count = 0

    bucket_count = 30
    buckets: dict[int, int] = {}
    for i in range(bucket_count):
        ts = int((thirty_min_ago + i * 60) // 60) * 60
        buckets[ts] = 0

    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        ts = entry["ts"]
                        if ts >= midnight:
                            jobs_today += 1
                        if ts >= now - 60:
                            jobs_last_minute += 1
                        if "duration" in entry:
                            total_duration += entry["duration"]
                            duration_count += 1
                        if ts >= thirty_min_ago:
                            bucket = int(ts // 60) * 60
                            if bucket in buckets:
                                buckets[bucket] += 1
                    except Exception:
                        pass
        except Exception:
            pass

    timeline = [{"ts": k, "count": v} for k, v in sorted(buckets.items())]
    avg_duration = round(total_duration / duration_count, 2) if duration_count else 0

    return {
        "jobs_per_minute": jobs_last_minute,
        "jobs_today": jobs_today,
        "avg_duration": avg_duration,
        "timeline": timeline,
    }


@app.get("/api/logs")
async def get_logs(lines: int = 200):
    if not WORKER_LOG.exists():
        return {"lines": []}
    result = subprocess.run(
        ["tail", f"-{lines}", str(WORKER_LOG)], capture_output=True, text=True
    )
    return {"lines": result.stdout.splitlines()}


async def _tail_log():
    if not WORKER_LOG.exists():
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        WORKER_LOG.touch()

    proc = await asyncio.create_subprocess_exec(
        "tail", "-n", "50", "-f", str(WORKER_LOG),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            yield f"data: {json.dumps(line.decode().rstrip())}\n\n"
    finally:
        proc.terminate()


@app.get("/api/logs/stream")
async def stream_logs():
    return StreamingResponse(_tail_log(), media_type="text/event-stream")


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>XBlock Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d3e;
    --text: #e2e8f0;
    --muted: #64748b;
    --accent: #6366f1;
    --green: #22c55e;
    --yellow: #eab308;
    --red: #ef4444;
    --blue: #3b82f6;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 14px; }
  header { display: flex; align-items: center; justify-content: space-between; padding: 16px 24px; border-bottom: 1px solid var(--border); }
  header h1 { font-size: 18px; font-weight: 600; letter-spacing: -0.3px; }
  .badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 500; }
  .badge::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
  .badge.running { background: rgba(34,197,94,.15); color: var(--green); }
  .badge.stopped { background: rgba(239,68,68,.15); color: var(--red); }
  .badge.starting { background: rgba(234,179,8,.15); color: var(--yellow); }
  .badge.unknown { background: rgba(100,116,139,.15); color: var(--muted); }
  main { max-width: 1200px; margin: 0 auto; padding: 24px; display: grid; gap: 20px; }
  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
  .card-label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; margin-bottom: 8px; }
  .card-value { font-size: 32px; font-weight: 700; letter-spacing: -1px; }
  .card-sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .controls { display: flex; gap: 10px; align-items: center; }
  .controls h2 { font-size: 14px; font-weight: 600; color: var(--muted); margin-right: 6px; }
  button { padding: 8px 16px; border-radius: 8px; border: 1px solid var(--border); background: var(--surface); color: var(--text); cursor: pointer; font-size: 13px; font-weight: 500; transition: background .15s, border-color .15s; }
  button:hover { background: var(--border); }
  button.start { border-color: rgba(34,197,94,.4); color: var(--green); }
  button.start:hover { background: rgba(34,197,94,.1); }
  button.stop { border-color: rgba(239,68,68,.4); color: var(--red); }
  button.stop:hover { background: rgba(239,68,68,.1); }
  button.restart { border-color: rgba(234,179,8,.4); color: var(--yellow); }
  button.restart:hover { background: rgba(234,179,8,.1); }
  #toast { display: none; position: fixed; bottom: 24px; right: 24px; padding: 10px 16px; border-radius: 8px; background: var(--surface); border: 1px solid var(--border); font-size: 13px; }
  .chart-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
  .chart-card h2 { font-size: 13px; font-weight: 600; color: var(--muted); margin-bottom: 16px; text-transform: uppercase; letter-spacing: .05em; }
  .chart-wrap { height: 140px; }
  .log-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
  .log-header { display: flex; align-items: center; justify-content: space-between; padding: 14px 20px; border-bottom: 1px solid var(--border); }
  .log-header h2 { font-size: 13px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
  #log-box { height: 340px; overflow-y: auto; padding: 12px 20px; font-family: "Cascadia Code", "Fira Code", "JetBrains Mono", monospace; font-size: 12px; line-height: 1.6; color: #94a3b8; }
  #log-box .line:last-child { color: var(--text); }
  #auto-scroll-toggle { font-size: 12px; color: var(--muted); cursor: pointer; }
  #auto-scroll-toggle:hover { color: var(--text); }
</style>
</head>
<body>
<header>
  <h1>XBlock Dashboard</h1>
  <span id="status-badge" class="badge unknown">Unknown</span>
</header>
<main>
  <div class="stats">
    <div class="card">
      <div class="card-label">Jobs Today</div>
      <div class="card-value" id="jobs-today">—</div>
      <div class="card-sub">since midnight</div>
    </div>
    <div class="card">
      <div class="card-label">Jobs / min</div>
      <div class="card-value" id="jobs-per-min">—</div>
      <div class="card-sub">last 60 seconds</div>
    </div>
    <div class="card">
      <div class="card-label">Avg Duration</div>
      <div class="card-value" id="avg-duration">—</div>
      <div class="card-sub">seconds per job</div>
    </div>
  </div>

  <div style="display:flex; align-items:center; gap:16px;">
    <div class="controls">
      <h2>Process</h2>
      <button class="start" onclick="control('start')">Start</button>
      <button class="stop" onclick="control('stop')">Stop</button>
      <button class="restart" onclick="control('restart')">Restart</button>
    </div>
  </div>

  <div class="chart-card">
    <h2>Throughput — last 30 min</h2>
    <div class="chart-wrap"><canvas id="throughput-chart"></canvas></div>
  </div>

  <div class="log-card">
    <div class="log-header">
      <h2>Worker Log</h2>
      <span id="auto-scroll-toggle" onclick="toggleAutoScroll()">Auto-scroll: ON</span>
    </div>
    <div id="log-box"></div>
  </div>
</main>
<div id="toast"></div>

<script>
let chart;
let autoScroll = true;

function toggleAutoScroll() {
  autoScroll = !autoScroll;
  document.getElementById('auto-scroll-toggle').textContent = `Auto-scroll: ${autoScroll ? 'ON' : 'OFF'}`;
}

function toast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.style.display = 'block';
  setTimeout(() => el.style.display = 'none', 3000);
}

async function control(action) {
  const res = await fetch(`/api/control/${action}`, { method: 'POST' });
  const data = await res.json();
  toast(data.output || data.detail || 'Done');
  setTimeout(refreshStatus, 1000);
}

async function refreshStatus() {
  const res = await fetch('/api/status');
  const data = await res.json();
  const badge = document.getElementById('status-badge');
  const s = (data.status || 'UNKNOWN').toLowerCase();
  badge.className = `badge ${s.includes('running') ? 'running' : s.includes('stop') ? 'stopped' : s.includes('start') ? 'starting' : 'unknown'}`;
  badge.textContent = data.status + (data.uptime ? ' · ' + data.uptime : '');
}

async function refreshMetrics() {
  const res = await fetch('/api/metrics');
  const data = await res.json();
  document.getElementById('jobs-today').textContent = data.jobs_today.toLocaleString();
  document.getElementById('jobs-per-min').textContent = data.jobs_per_minute;
  document.getElementById('avg-duration').textContent = data.avg_duration + 's';

  const labels = data.timeline.map(b => {
    const d = new Date(b.ts * 1000);
    return d.getHours().toString().padStart(2,'0') + ':' + d.getMinutes().toString().padStart(2,'0');
  });
  const counts = data.timeline.map(b => b.count);

  if (!chart) {
    const ctx = document.getElementById('throughput-chart').getContext('2d');
    chart = new Chart(ctx, {
      type: 'bar',
      data: { labels, datasets: [{ data: counts, backgroundColor: 'rgba(99,102,241,.6)', borderColor: 'rgba(99,102,241,1)', borderWidth: 1, borderRadius: 3 }] },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#64748b', maxTicksLimit: 8 }, grid: { color: '#2a2d3e' } },
          y: { ticks: { color: '#64748b', precision: 0 }, grid: { color: '#2a2d3e' }, beginAtZero: true },
        }
      }
    });
  } else {
    chart.data.labels = labels;
    chart.data.datasets[0].data = counts;
    chart.update('none');
  }
}

function appendLog(line) {
  const box = document.getElementById('log-box');
  const el = document.createElement('div');
  el.className = 'line';
  el.textContent = line;
  box.appendChild(el);
  if (box.children.length > 500) box.removeChild(box.firstChild);
  if (autoScroll) box.scrollTop = box.scrollHeight;
}

async function initLogs() {
  const res = await fetch('/api/logs?lines=80');
  const data = await res.json();
  data.lines.forEach(appendLog);

  const evtSource = new EventSource('/api/logs/stream');
  evtSource.onmessage = e => appendLog(JSON.parse(e.data));
}

async function tick() {
  await Promise.all([refreshStatus(), refreshMetrics()]);
}

tick();
initLogs();
setInterval(tick, 5000);
</script>
</body>
</html>"""


@app.get("/")
async def index():
    return HTMLResponse(HTML)
