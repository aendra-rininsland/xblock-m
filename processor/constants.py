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


# Per-class operating points.
#
# One global threshold across twelve classes assumes they are equally
# well-separated, and they are not. At 0.8 the model produced ZERO predictions
# for most classes despite having clear ranking signal in them -- average
# precision well above chance while precision and recall were both 0.00. The
# ranking was fine; the gate was in the wrong place.
#
# Fitted per class on the validation split by:
#     python evaluate.py --tune-thresholds --write-constants thresholds.py
# which picks the lowest threshold still meeting a precision target, since
# anything higher costs recall for nothing.
#
# A class absent from this map falls back to THRESHOLD. That is deliberate: a
# threshold fitted to three validation images is noise, so classes without
# enough support are better left at a conservative default.
CLASS_THRESHOLDS: dict[str, float] = {}


def threshold_for(label: str) -> float:
    """The operating point for one class."""
    return CLASS_THRESHOLDS.get(label, THRESHOLD)
