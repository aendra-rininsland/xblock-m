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

export const queue = new Queue("xblock", {
  connection: { host: REDIS_HOST },
});

// Separate low-traffic client used only for heartbeat writes.
const redis = new Redis({ host: REDIS_HOST, lazyConnect: true });
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
