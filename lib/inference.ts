import { CommitCreateEvent } from "@skyware/jetstream";
import { Job } from "bullmq";

// Uncomment the following and comment out the line after to use another pretrained model
// import { pipeline } from "@xenova/transformers";
import { pipeline } from "./pipeline";

// Allocate a pipeline
const model = pipeline(
  "multi-label-image-classification", // Probably change to 'image-classification'
  "howdyaendra/microsoft-swinv2-small-patch4-window16-256-finetuned-xblockm" // e.g. 'Xenova/vit-base-patch16-224'
);

export default async function (
  job: Job<CommitCreateEvent<"app.bsky.feed.post">>
) {
  try {
    await job.log("Start processing job");
    const pipeline = await model;

    if (job.data.commit.record.embed?.$type === "app.bsky.embed.images") {
      const urls = job.data.commit.record.embed.images.map(
        (d) =>
          `https://cdn.bsky.app/img/feed_fullsize/plain/${job.data.did}/${d.image.ref.$link}@jpeg`
      );

      const results = await pipeline(urls);

      // What you do with the results is up to you...

      const filtered = results.filter((p) =>
        p.some((d) => d.score > 0.75 && d.label !== "negative")
      );
      if (filtered.length > 0) {
        console.log(urls, filtered);
      }
    }
  } catch (e) {
    console.error(e);
  }
}