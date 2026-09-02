/**
 * @file
 * Jetstream firehose consumer
 */

import { CommitCreateEvent, Jetstream } from "@skyware/jetstream";
import WebSocket from "ws";
import debug from "debug";
import { Queue } from "bullmq";
import Redis from "ioredis";

const REDIS_HOST = process.env.REDIS_HOSTNAME ?? "redis";

// Undefined when unset, which ioredis treats as "no auth" -- so this stays
// compatible with a Redis that has no requirepass. The migrated queue box does
// set one: Redis there binds to loopback and WireGuard only and 6379 is
// firewalled, but the old box's public 6379 behind an IP allowlist is exactly
// the arrangement that quietly stopped working whenever the home IP changed,
// so the password is the belt to the tunnel's braces.
const REDIS_PASSWORD = process.env.REDIS_PASSWORD || undefined;

export const queue = new Queue("xblock", {
  connection: { host: REDIS_HOST, password: REDIS_PASSWORD },
});

// Separate low-traffic client used only for heartbeat writes.
const redis = new Redis({ host: REDIS_HOST, password: REDIS_PASSWORD, lazyConnect: true });
redis.on("error", (e) => console.error("[firehose] redis error:", e));

const log = debug("xblock:firehose");

// ── Dead man's switch ────────────────────────────────────────────────────────
// Bluesky emits thousands of app.bsky.feed.post events per minute globally.
// If we go 5 minutes without seeing any, the Jetstream WebSocket has silently
// failed. Exit so pm2 can restart the process.
const IDLE_TIMEOUT_MS  = 5 * 60 * 1000; // 5 minutes
const STARTUP_GRACE_MS = 2 * 60 * 1000; // don't check until fully warmed up
const startedAt = Date.now();
let lastEventAt = Date.now();

setInterval(() => {
  if (Date.now() - startedAt < STARTUP_GRACE_MS) return;
  const idleMs = Date.now() - lastEventAt;
  if (idleMs > IDLE_TIMEOUT_MS) {
    console.error(
      `[firehose] No events in ${Math.round(idleMs / 1000)}s — exiting for pm2 restart`
    );
    process.exit(1);
  }
}, 30_000);

// ── Jetstream ────────────────────────────────────────────────────────────────

const jetstream = new Jetstream({
  ws: WebSocket,
  wantedCollections: ["app.bsky.feed.post"],
});

// Jetstream is an EventEmitter, so an 'error' event with no listener is not a
// logged warning -- Node rethrows it as an uncaught exception and the process
// dies. Every transient Jetstream websocket blip was therefore killing the
// consumer: 31 restarts in 41 hours as measured on 2 September 2026, each one a
// gap in firehose consumption.
//
// The reconnect already exists. @skyware/jetstream wraps ReconnectingWebSocket,
// which backs off and reconnects on its own -- but only if the process survives
// long enough to let it. Handling the event is the whole fix.
//
// This does not weaken the dead man's switch above: a socket that reconnects
// but delivers nothing still trips the 5-minute idle check and still exits for
// pm2. What changes is that a blip is no longer indistinguishable from death.
jetstream.on("error", (e) => console.error("[firehose] jetstream error:", e));

class Batch {
  items: CommitCreateEvent<"app.bsky.feed.post">[] = [];
  max = 5;

  constructor(max: number) {
    this.max = max;
  }

  add(item: CommitCreateEvent<"app.bsky.feed.post">) {
    this.items.push(item);
    if (this.items.length >= this.max) {
      const payload = [...this.items];
      this.items = [];
      queue
        .add(new Date().toISOString(), payload, {
          lifo: true,
          removeOnComplete: true,
          removeOnFail: true,
        })
        .then(() => {
          // Heartbeat key read by the local watchdog to detect a dead firehose.
          redis
            .set("xblock:firehose:last_enqueue_ts", Date.now().toString(), "EX", 86400)
            .catch(() => {});
          log(payload);
        })
        .catch((e) => console.error(e));
    }
  }
}
const batch = new Batch(Number(process.env.BATCH_SIZE ?? 1));

jetstream.onCreate("app.bsky.feed.post", (event) => {
  lastEventAt = Date.now(); // reset idle timer on ANY post, before image filter
  if (
    event.commit.record.embed?.$type === "app.bsky.embed.images" &&
    event.commit.record.langs?.includes("en")
  ) {
    batch.add(event);
  }
});

jetstream.start();

console.log("Firehose consumer running...");
