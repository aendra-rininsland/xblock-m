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
| `multi-candidate` | two classes both scoring ≥ 0.35 | nested screenshots — the case the single-label corpus could never express, and which random sampling essentially never turns up |

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

Check for truncated downloads, and purge them so a fresh harvest refetches:

```bash
python harvest.py verify
python harvest.py verify --repair
```

A partial JPEG still *decodes* — it renders the scanlines that arrived and fills
the rest — so a truncated download lands in the corpus looking like a real image.
`verify` catches them structurally (a JPEG must end with its EOI marker). It
refuses to delete anything carrying a human label.

Inspect, and emit a review queue:

```bash
python harvest.py stats
python harvest.py export --split unassigned --out review.jsonl
```

`--db` defaults to `$PIPELINE_DB` if set, so this can share the database
`processor/worker.py` already writes model scores into.

## Review

```bash
python review.py --db data/pipeline.db --images data/images
# http://127.0.0.1:8081
```

Two modes, because the work has two shapes:

**Sweep** — a grid of thumbnails. Click the few that *are* screenshots; everything
left unclicked is recorded as `negative` on submit. This is how ~3,000 negatives
get labelled in well under an hour. Thumbnails are top-anchored rather than
centre-cropped: platform chrome lives along the top edge, and centre-cropping a
tall screenshot to a square throws away the part you are judging.

**Detail** — one image at a time, multi-select by keyboard. For whatever the
sweep flagged, and for the `fired` and `uncertain` buckets where the judgement
matters. Model scores are shown inline per class.

| key | |
|---|---|
| `1`–`9`, `q`, `w` | platform classes, ordered by corpus frequency |
| `0` | negative — exclusive, clears the rest |
| `↵` | confirm page (sweep) / save and advance (detail) |
| `s` `u` `←` `→` | skip, undo, navigate |

Multi-select is the point: a nested screenshot fires more than one class, and
that is the case the old ImageFolder corpus could not express.

Nothing is auto-labelled. `negative` is only ever inferred from an explicit
human submit of a sweep page.

## Importing the existing corpus

```bash
python import_corpus.py            # dry run (the default)
python import_corpus.py --apply
```

Pulls the ImageFolder dataset in, applying the same class surgery as the training
notebook (drop `news`, rename `altright` → `truthsocial`).

**A CID in two folders means both labels**, so every file is read rather than
deduplicated on first sight. 37 of 1,581 distinct images are filed twice — but
only 6 are genuine co-occurrence (0.38%). The other 31 pair a platform with
`negative`, which is a contradiction rather than a co-occurrence (29 are
`discord + negative`, which looks like one bulk misfile). Those are resolved from `conflict_resolutions.json` where a human has recorded a
decision, and otherwise import with **no label** and go to the review queue —
never guessed.

The 29 `discord + negative` images were reviewed by hand: 27 are Discord, 2 are
genuinely negative. That means the `negative` class was ~15% Discord screenshots,
which matters more than the count suggests, since negatives are what govern the
false-positive rate. The decisions live in the repo so they survive rebuilding
the manifest from scratch.

**Imported labels are "at least this", not "exactly this."** Double-filing is a
lower bound on co-occurrence, not a measurement: a nested screenshot filed once
under its dominant platform is indistinguishable from a single-platform one. That
is why imported rows keep `review_state='imported'` and stay re-reviewable rather
than being marked done — including the 29 conflict resolutions, which settled
which folder was right and said nothing about whether a second platform appears. That matters under `BCEWithLogitsLoss`, where an unrecorded positive
becomes an explicit zero in the target — training the model to suppress the very
co-occurrence you want it to learn.

So imported rows get `source='imagefolder-v1'` and `review_state='imported'`,
never `human`/`done`. They stay out of the default review queue but the
`imported (re-review)` bucket filter surfaces them, and a human label supersedes
the imported one.

## Importing moderation signal from Ozone

```bash
export OZONE_DB_URL=postgres://user:pass@host/ozone
python import_ozone.py --bot-did did:plc:...     # dry run
python import_ozone.py --bot-did did:plc:... --apply
```

Ozone holds the moderation record, not images — but `moderation_event` carries
`subjectBlobCids`, so images are recoverable from the CDN by CID.

| bucket | what it is |
|---|---|
| `ozone-false-positive` | a human negated a label the model applied — a confirmed FP with a moderator's judgement attached |
| `ozone-appeal` | a report with an appeal `reasonType` — the strongest FP evidence there is |
| `ozone-reported` | user-submitted report |
| `ozone-human` | a moderator applied this label by hand |
| `ozone-model` | the labeller fired; unverified |

**Nothing from Ozone is written as `source='human'`** unless a human moderator
actually did it. Model labels are the model's own predictions: training on them
entrenches the current decision boundary instead of correcting it, and drift *is*
a decision-boundary problem. They arrive as candidates for review.

Ozone also has **almost no negatives** — everything in it either fired or was
complained about. It is not a sample of the firehose, so `harvest.py` stays
necessary for the corpus's largest gap.

`--bot-did` is required: `createdBy` is what separates the model's own labels
from a moderator's, and it works across the whole history. `modTool.meta.isAutomated`
is Ozone's own field for this and `moderate.py` now sets it, but older events
predate it.

## Splits

```bash
python assign_splits.py            # dry run (the default), with a per-class report
python assign_splits.py --apply
```

Assigned **once**, in the manifest, and **never reassigned** — that is what keeps
a frozen evaluation set frozen as the corpus grows. The previous notebook split
inside `train()` and pointed its test set at the whole corpus, which is why its
`TEST AUROC 0.983` read above `valid 0.959`.

Stratification is the usual greedy approximation of iterative stratification
(labels are sets, so ordinary stratification doesn't apply), plus a floor pass:
at 80/10/10 a six-image class rounds to 0.6 expected in each of val and test, and
greedy assignment leaves one of them empty — which makes the class *invisible* to
evaluation rather than merely noisy. One image per split is reserved for every
class that can spare it. Read the per-class report; with classes this small it
matters more than the method.

## Training from the manifest

```python
from dataset import load_manifest_dataset
ds = load_manifest_dataset(DB, IMAGES, split="train")
```

`labels` arrives already multi-hot, replacing the notebook's

```python
labels = torch.tensor([[x] for x in batch['label']])
batch['labels'] = nn.functional.one_hot(labels, num_classes).sum(dim=1)
```

which assumes exactly one integer label per image and so cannot express a
co-occurrence however the model is trained.

### Multi-label coverage

```bash
python dataset.py --co-occurrence
```

Multi-label is a design property of the system — the head is sigmoid and a
screenshot of a screenshot genuinely carries two labels. The data doesn't have to
justify that. But a class the model never sees co-occurring **cannot learn to
co-occur**, so this reports coverage: which pairs have been observed, and which
classes have no co-occurrence example at all. Those can only ever be predicted
alone, and the `multi-candidate` harvest bucket targets exactly that gap.

## Schema

- `images` — one row per distinct image. `cid` is a content hash, so byte-identical
  re-uploads deduplicate for free and harvests are resumable.
- `labels` — `(cid, label, source)`. **Multi-label by construction.** `source`
  separates a human decision from a model guess or an Ozone appeal.
- `splits` — kept apart from `labels` so relabelling an image can never silently
  move it between train and eval. A frozen eval set only stays frozen if its
  membership lives somewhere relabelling does not touch.
- `review_state` — review progress, kept apart from `labels` so "confirmed
  negative" and "not looked at yet" stay distinguishable. Without it an image
  with no labels is ambiguous and a 3,000-image sweep cannot be resumed.
- `model_scores` — shape-compatible with what `worker.py` already writes.

## Notes

Harvested images arrive **unlabelled by design**. Nothing here guesses a label;
`export` produces a queue for review.

Sampling mirrors the production filter in `lib/firehose.ts` exactly — image
embeds, English-tagged, creates only — so the harvested distribution matches what
the model actually sees rather than the whole network.

Preprocessing for `--score` comes from `processor/preprocessing.py`, shared with
the worker so collection and serving cannot drift apart.
