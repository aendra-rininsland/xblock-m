"""
The map between model class names and the label values published to subscribers.

Free of heavy imports so both the serving path (moderate.py) and the pipeline
tooling can use it without dragging in atproto or torch.
"""
from __future__ import annotations

# Class names no longer in the training set. Retained so a rollback to an older
# checkpoint cannot publish something unexpected, and so historical Ozone events
# can still be read back.
RETIRED_CLASSES = {"altright", "news", "newsmedia"}

# Model class name -> the label value published to subscribers.
#
# Kept explicit rather than interpolated as f"{label}-screenshot" so the two can
# be versioned independently. Renaming a class in the training set must not
# silently change what subscribers receive: anyone filtering on the old value
# would simply stop matching, with nothing in the logs to say why.
#
# None means "the model may predict this, but nothing is published".
LABEL_VALUES: dict[str, str | None] = {
    # `altright` was renamed to `truthsocial` in the training set -- it is a
    # platform class (Truth Social's UI), not an ideological judgement. The
    # published value stays the legacy one until subscribers have migrated;
    # flip it to "truthsocial-screenshot" once they have.
    "truthsocial": "altright-screenshot",
    "bluesky":     "bluesky-screenshot",
    "discord":     "discord-screenshot",
    "facebook":    "facebook-screenshot",
    "fediverse":   "fediverse-screenshot",
    "instagram":   "instagram-screenshot",
    "ngl":         "ngl-screenshot",
    "reddit":      "reddit-screenshot",
    "threads":     "threads-screenshot",
    "tumblr":      "tumblr-screenshot",
    "twitter":     "twitter-screenshot",

    # "not a screenshot" is the absence of a label, never a label of its own.
    "negative":    None,

    # Retired classes. Retained so that rolling back to an older checkpoint
    # cannot start publishing something unexpected.
    "altright":    "altright-screenshot",
    "news":        None,
    "newsmedia":   None,
}


def _build_inverse() -> dict[str, str]:
    """Published value -> CURRENT model class.

    Two classes can publish the same value: `truthsocial` and the retired
    `altright` both emit "altright-screenshot". Reading an old Ozone event back
    has to resolve to the current name, so live classes are applied last and win.
    """
    inverse: dict[str, str] = {}
    for name in sorted(LABEL_VALUES, key=lambda n: n in RETIRED_CLASSES, reverse=True):
        value = LABEL_VALUES[name]
        if value is not None:
            inverse[value] = name
    return inverse


VALUE_TO_CLASS = _build_inverse()


def model_class(published_value: str) -> str | None:
    """Map a label value seen in Ozone back to a current model class, or None."""
    return VALUE_TO_CLASS.get(published_value)
