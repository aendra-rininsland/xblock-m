/**
 * @file
 * Worker process. This spawns the inference daemons.
 */

import { Worker } from "bullmq";

export const worker = new Worker(
  "xblock",
  `${process.cwd()}/lib/inference.ts`,
  {
    connection: {
      host: process.env.REDIS_HOSTNAME ?? "redis",
      // undefined when unset, so a Redis without requirepass still works
      password: process.env.REDIS_PASSWORD || undefined,
    },
    concurrency: Number(process.env.WORKER_CONCURRENCY ?? 2),
  }
);
