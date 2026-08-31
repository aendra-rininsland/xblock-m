# pipeline

Data collection for retraining. Samples the live firehose into a SQLite manifest
where an image carries a **set** of labels.

## Why

Two problems with the current corpus that no amount of training fixes:

**It cannot express co-occurrence.** The training set is an ImageFolder tree —
one directory per class, so exactly one label per image. The nested-screenshot
case (a Bluesky post quoting a Twitter screenshot, firing both) is a stated goal
of the project that the data structurally cannot teach.

**It is short of negatives.** 194 negatives against 1,140 positives, roughly
0.17×, deployed on a stream that is overwhelmingly not-a-screenshot. A model
trained on a balanced prior and served on a skewed one over-fires, and false
positives are the failure people actually report. Reaching 3× positives needs
about 3,200 more — which the firehose gives away free.

## Buckets

Harvesting only what the model already scores low would fill the corpus with easy
negatives that teach nothing. So:

| bucket | what it is | why you want it |
|---|---|---|
| `random` | unbiased sample of the production stream | the true prior — what a frozen eval set needs and a curated corpus cannot give you |
| `fired` | model scores ≥ threshold on some class | candidate false positives: the negatives actually worth labelling |
| `uncertain` | top non-negative score in 0.35–0.75 | where the decision boundary is, so where new labels move it most |

`random` needs no model. `fired` and `uncertain` need `--score`.

## Usage

Collect negatives — no GPU, no model, runs anywhere:

```bash
python harvest.py harvest --target 4000 --rate 0.05
```

Mine current false positives (needs the processor/ deps):

```bash
python harvest.py harvest --target 500 --bucket fired
```

Inspect, and emit a review queue:

```bash
python harvest.py stats
python harvest.py export --split unassigned --out review.jsonl
```

`--db` defaults to `$PIPELINE_DB` if set, so this can share the database
`processor/worker.py` already writes model scores into.

## Schema

- `images` — one row per distinct image. `cid` is a content hash, so byte-identical
  re-uploads deduplicate for free and harvests are resumable.
- `labels` — `(cid, label, source)`. **Multi-label by construction.** `source`
  separates a human decision from a model guess or an Ozone appeal.
- `splits` — kept apart from `labels` so relabelling an image can never silently
  move it between train and eval. A frozen eval set only stays frozen if its
  membership lives somewhere relabelling does not touch.
- `model_scores` — shape-compatible with what `worker.py` already writes.

## Notes

Harvested images arrive **unlabelled by design**. Nothing here guesses a label;
`export` produces a queue for review.

Sampling mirrors the production filter in `lib/firehose.ts` exactly — image
embeds, English-tagged, creates only — so the harvested distribution matches what
the model actually sees rather than the whole network.

Preprocessing for `--score` comes from `processor/preprocessing.py`, shared with
the worker so collection and serving cannot drift apart.
