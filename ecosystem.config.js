module.exports = {
  apps: [
    {
      name: "xblock-firehose",
      script: "lib/firehose.ts",
      interpreter: "./node_modules/.bin/tsx",
      interpreter_args: "-r dotenv/config",
      autorestart: true,
      // Exponential backoff: 1s → 2s → 4s … capped at 15s by pm2.
      // The dead man's switch already adds a ~5 min delay before exiting,
      // so rapid flapping is unlikely; this is just a safety net.
      exp_backoff_restart_delay: 1000,
      // After this many restarts pm2 marks the app as "errored" and stops.
      // Set high because transient Jetstream outages can cause several restarts.
      max_restarts: 20,
      // Must run for at least 30s before a restart counts as a success.
      min_uptime: "30s",
      watch: false,
    },
  ],
};
