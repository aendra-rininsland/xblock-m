import os

THRESHOLD = 0.8

# Jobs older than this are no longer worth processing.
#
# The LIFO queue is deliberate: recent posts are prioritised and the backlog is
# worked through during quiet periods. But a Bluesky post has a half-life of
# hours, so a label applied days late reaches almost nobody -- the post has long
# since scrolled past every timeline that would have filtered on it. Those jobs
# are pure cost: they hold gigabytes of Redis and buy nothing.
#
# This is an age bound, deliberately not a depth cap. A depth cap would discard
# the NEWEST work during a burst, which is exactly the diurnal buffering the LIFO
# design exists to provide. An age bound only discards work whose value has
# already expired.
MAX_JOB_AGE_HOURS = float(os.getenv("MAX_JOB_AGE_HOURS", "48"))
MAX_JOB_AGE_MS = MAX_JOB_AGE_HOURS * 3600 * 1000
