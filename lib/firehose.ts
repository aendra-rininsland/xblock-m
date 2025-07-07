/**
 * @file
 * Jetstream firehose consumer
 */

import { CommitCreateEvent, Jetstream } from "@skyware/jetstream";
import WebSocket from "ws";
import debug from "debug";
import { Queue } from "bullmq";

export const queue = new Queue("xblock", {
  connection: {
    host: process.env.REDIS_HOSTNAME ?? "redis",
  },
});

const log = debug("xblock:firehose");

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
      queue
        .add(new Date().toISOString(), [...this.items], {
          lifo: true,
          removeOnComplete: true,
          removeOnFail: true,
        })
        .catch((e) => console.error(e));
      log(this.items);
      this.items = [];
    }
  }
}
const batch = new Batch(Number(process.env.BATCH_SIZE ?? 1));

jetstream.onCreate("app.bsky.feed.post", (event) => {
  if (
    event.commit.record.embed?.$type === "app.bsky.embed.images" &&
    event.commit.record.langs?.includes("en")
  ) {
    batch.add(event);
  }
});

jetstream.start();

console.log("Firehose consumer running...");
