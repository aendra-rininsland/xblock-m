import express from "express";
import { Queue } from "bullmq";
import { createBullBoard } from "@bull-board/api";
import { BullMQAdapter } from "@bull-board/api/bullMQAdapter";
import { ExpressAdapter } from "@bull-board/express";

const queueMQ = new Queue("xblock", {
  connection: {
    host: process.env.REDIS_HOSTNAME ?? "redis",
    // undefined when unset, so a Redis without requirepass still works
    password: process.env.REDIS_PASSWORD || undefined,
  },
});

const serverAdapter = new ExpressAdapter();
serverAdapter.setBasePath("/admin/queues");

createBullBoard({
  queues: [new BullMQAdapter(queueMQ)],
  serverAdapter: serverAdapter,
});

const app = express();

app.use("/admin/queues", serverAdapter.getRouter());

// other configurations of your server

app.listen(3000, () => {
  console.log("Running on 3000...");
  console.log("For the UI, open http://localhost:3000/admin/queues");
  console.log("Make sure Redis is running on port 6379 by default");
});
