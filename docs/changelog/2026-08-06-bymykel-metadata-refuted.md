# The ByMykel bundle is wired in behind a flag, and the model does not use it

> **Resolved 2026-08-06, later the same evening.** This entry originally closed as
> *inconclusive* — the paired retrain neither confirmed nor refuted the bundle. A
> 700K-budget run settled it: the forecaster's **own** in-model permutation test
> reports that shuffling the ByMykel group costs **−0.24 / −0.69 / −0.06 / +0.79pp**
> at 3/7/14/30d against `price_technicals` at **+1.15 to +12.18pp, p=0.0000**. The
> model is not using the columns, so the paired diffs below are training
> nondeterminism rather than a feature effect. **Refuted; the flag stays off
> permanently.** See § The 700K run, which settles it.

**Date:** 2026-08-06
**Change:** `models/forecaster.py` gains a default-**off** `BYMYKEL_METADATA` flag,
a `bymykel_metadata` feature group, `_add_bymykel_metadata_features`, and an artifact
guard. New `scripts/paired_retrain_bymykel.py`. New
`tests/test_bymykel_metadata_wiring.py` (21 tests). **The flag stays off.**
**Follows:** `2026-08-06-bymykel-metadata-ingest.md`, which measured the bundle at
+1.26pp (7d) and +3.72pp (30d) on the held-out-item CV harness.

The question this answers is whether that CV result survives the production retrain
path. **It does not — not because it measured negative, but because the instrument
cannot resolve it.** The rule passes on point estimates and the evidence behind those
point estimates does not hold up.

## The verdict

Two runs of the *same* configuration, and a permuted placebo arm.

Paired `classifier_accuracy_ge1` (the ≥$1 cohort the production headline reports),
folds matched on identical `(train_start, train_end, val_start, val_end)`,
treatment − control in percentage points:

| h | folds | run 1 | run 2 | run 2 placebo | run 2 MDE(80%) |
|---|---:|---|---|---|---:|
| 3d | 9 | +1.52 [−0.24, +3.28] | +0.66 [−0.95, +2.26] | −0.38 [−0.99, +0.23] | 2.29 |
| 7d | 8 | **+3.57 [+1.85, +5.30]** | **+1.91 [+0.51, +3.32]** | **+1.20 [+0.44, +1.96]** | 2.01 |
| 14d | 8 | +1.74 [−1.35, +4.82] | +2.84 [−1.22, +6.89] | +0.39 [−5.27, +6.04] | 5.80 |
| 30d | 8 | +0.48 [−3.90, +4.85] | +2.46 [−4.58, +9.51] | +0.41 [−4.20, +5.02] | 10.07 |

Both runs clear the pre-registered bar — *> +1pp at two or more horizons, not worse
than −1pp at any* — and **they clear it at different horizons**: run 1 at 3d/7d/14d,
run 2 at 7d/14d/30d. A rule satisfied by a different set of horizons each time is
being satisfied by noise.

Three separate reasons not to act on it:

**1. The placebo is significantly positive at the only horizon that is significant.**
Permuting the metadata across items — whole rows shuffled against `item_slug`, so
each column keeps its distribution and only the item→metadata assignment is
destroyed — buys **+1.20pp [+0.44, +1.96] at 7d**. Treatment at 7d is +1.91pp. Read
against the placebo rather than against zero, the 7d effect is ~+0.7pp and well
inside the noise. This is the failure mode `ab-fold-count-floor` records: at ~7 folds
a permuted placebo has already read significantly positive on this project's designs,
and this harness runs 8–9.

**2. The instrument is not reproducible at the effect size.** Between the two runs the
*control* arm moved on its own — 7d control 48.38% → 47.16%, 3d 49.31% → 49.98% — with
identical code, identical data and identical settings, control running first in both.
The paired diffs swing by up to 1.7pp between runs, which is larger than the effect
being measured. A single run of this harness cannot evaluate a ±1pp rule.

**3. Every other horizon is below its own MDE.** 3d, 14d and 30d have MDEs of
2.29–10.07pp against diffs of +0.66 to +2.84pp. Their CIs all include zero. The rule
counts them as wins because it is written on point estimates; the instrument cannot
tell them from zero.

**The flag therefore stays off**, and the CV result in
`2026-08-06-bymykel-metadata-ingest.md` stands as what it always was — a held-out CV
number on an 870-item universe, not a production result.

## Why the two instruments disagree

They are not measuring the same thing, and the gap is mostly cohort and scale:

* **870 items vs ~99.** The CV harness trains on 500 items and evaluates on 150
  held-out ones. The production path at `DEFAULT_TRAIN_FEATURE_ROWS = 100_000` sees
  ~99 items (`training-uses-133-item-subsample`). Nine metadata columns describing
  item identity have far less to say across 99 items than across 500.
* **26 folds vs 8–9.** The CV harness's fold count is what gives it MDEs of
  0.20–0.75pp. The production path's 8–9 folds give 2.0–10.1pp. The measured effect
  sits in the gap between the two.
* **Held-out items vs time-separated folds.** The CV harness scores generalisation to
  items no arm trained on, which is the regime item metadata should most help.
  Production trains on most of what it serves.

None of this says the CV result was wrong. It says the production path as currently
configured cannot see an effect of that size, which is a statement about the
instrument.

## What was wired, and the traps found while wiring it

**A dedicated feature group, not an existing one.** `_feature_group` routes the nine
columns to `bymykel_metadata` by exact name. The obvious alternative — adding
`item_identity` / `item_metadata` / `temporal` to the allowlist — would have admitted
the 2026-07-24 ablation's 85 refuted features alongside the nine measured ones and
made any result unattributable.

**`item_age_meta_days`, not `item_age_days`.** The name was already taken by a
different quantity: `_add_temporal_features` computes observation date − first date
in the price frame, where this is observation date − catalogue first-sale date. Same
units, different meaning. Reusing the name would have silently overwritten one with
the other, and the A/B harness does exactly that overwrite in its own frame.

**The nullable-`Int64` trap.** `_select_feature_cols` keeps a column only when its
dtype is in `(float64, float32, int64, int, float)`. The ingest's source columns are
pandas nullable `Int64`, which is none of those — left uncast, the columns are
dropped silently and the treatment arm measures a clean, entirely spurious null. Cast
to float64 at join time, and pinned by a test.

**The correlation prune takes two of the nine.** Production reaches 33 → **40**
features, not 42: `is_meta_stattrak` and `is_meta_souvenir` are pruned above the 0.95
threshold as near-duplicates of the `is_stattrak` / `is_souvenir` that
`_add_item_identity_features` already parses from the name. This is correct behaviour
and a real difference from the CV harness, which carried all nine.

**The artifact guard is separate from the version bump.**
`MODEL_ARTIFACT_VERSION` stays at 5. With the flag off the feature set is
byte-identical to today, so bumping would force every checkout into a needless
retrain for a change none of them enabled. Instead the artifact records
`bymykel_metadata`, and `_check_artifact_version` refuses to load across a mismatch —
a booster trained with the columns and served without them is the train/serve
mismatch the v3 and v4 bumps exist to prevent. An artifact predating the key reads as
disabled, which is true of every existing one.

## The 700K run, which settles it

Run at `TRAIN_FEATURE_ROWS=700_000` — **646 items** in the frame instead of ~99, the
regime the CV result was measured in, and the one thing identified above as able to
address the disagreement. ~15 min per arm.

Paired diffs looked like the best result yet:

| h | folds | control | treatment | diff | 95% CI | MDE(80%) |
|---|---:|---|---|---|---|---:|
| 3d | 9 | 50.64% | 51.59% | +0.94 | [+0.40, +1.49] | 0.78 |
| 7d | 8 | 49.49% | 50.35% | +0.86 | [−0.59, +2.32] | 2.08 |
| 14d | 8 | 49.41% | 53.30% | **+3.89** | [+1.58, +6.19] | 3.30 |
| 30d | 8 | 51.50% | 55.81% | +4.31 | [−1.59, +10.21] | 8.44 |

3d and 14d clear both their CIs and their MDEs. Taken alone this reads as the first
genuinely positive production-path result for anything in months.

**It is not one.** The forecaster already runs a per-group permutation test during
training (`_validate_feature_groups`), and in the **treatment** arm — real,
unpermuted metadata — it reports:

| group | 3d | 7d | 14d | 30d |
|---|---|---|---|---|
| `price_technicals` | +9.10 | +11.51 | +5.19 | +10.76 (all p=0.0000, PASS) |
| `bymykel_metadata` | −0.24 | −0.69 | −0.06 | +0.79 |

Shuffling the bundle costs **nothing**, and at 3d/7d/14d it is *negative* — the model
scores marginally better with the columns randomised. Three of four horizons WARN;
only 30d passes, at +0.79pp. A feature group worth +3.89pp at 14d would lose ~3.89pp
when shuffled. This one loses 0.06pp.

So the paired diff and the permutation test disagree, and the permutation test is the
better instrument: it is **within-run**, so it cannot be contaminated by the
training nondeterminism that the paired diff measures across two separate trainings.

### The placebo arm was never a placebo, and what it accidentally measured

The permuted arm finished with **32 features and zero bundle columns — identical to
control.** `_validate_feature_groups` correctly scored the permuted group at −0.02pp
and `prune_failed_groups=True` deleted it outright. The arm therefore trained the
control feature set.

That makes its diffs a direct measurement of **run-to-run nondeterminism between two
identical configurations**, which is more useful here than the placebo would have
been:

| h | 100K "placebo" | 700K "placebo" |
|---|---|---|
| 3d | −0.38 [−0.99, +0.23] | −0.00 [−0.09, +0.09] |
| 7d | **+1.20 [+0.44, +1.96]** | −0.90 [−1.85, +0.05] |
| 14d | +0.39 [−5.27, +6.04] | +0.69 [−0.25, +1.63] |
| 30d | +0.41 [−4.20, +5.02] | +0.35 [−1.13, +1.83] |

At 100K, two identical configurations differ by **+1.20pp with a CI excluding zero**.
That is the instrument's noise floor, measured, and it is larger than most effects
this project tries to detect. It shrinks at 700K but does not vanish.

**A useful by-product:** `prune_failed_groups` distinguished real metadata from
permuted metadata unaided — it kept 7 columns in treatment and deleted all 9 in the
permuted arm. The mechanism works; there is simply nothing for it to keep.

## The lesson for the next feature experiment

**Read `_validate_feature_groups` before reading any paired retrain.** It is already
computed on every training run, it costs nothing extra, it is within-run so
nondeterminism cannot contaminate it, and it answers the question the paired retrain
only approaches: *does the model use this feature at all?* Had it been read first,
this line of work would have closed after one 15-minute run instead of four.

The paired retrain remains the right instrument for a change that alters *labels*,
*weights* or *splits* — where there is no feature group to permute. It is the wrong
first instrument for a change that adds columns.

Second: **two runs of an identical configuration differ by up to 1.20pp with a CI
excluding zero** at the 100K budget. Any production-path result below roughly that
magnitude, in either direction, is unreadable from a single run. That number belongs
next to the MDE in any future design.

## What was deliberately not done

* **The flag was not enabled**, no production retrain was run, and
  `models/saved_models/` was not touched — both arms train into scratch dirs.
* **`MODEL_ARTIFACT_VERSION` was not bumped.** See above.
* **No repeat runs at 700K.** One run, settled by the permutation test rather than by
  repetition. Repeats would only sharpen a paired diff now known to be measuring
  nondeterminism.
* **The flag was left in the codebase rather than reverted.** The ingest, the join and
  the artifact guard are all correct and tested; only the hypothesis failed. Removing
  them would cost the next person the rebuild, and the parquet is useful metadata
  regardless. Nothing reads it with the flag off.

## Related

* `docs/changelog/2026-08-06-bymykel-metadata-ingest.md` — the ingest and the CV result
* `docs/changelog/2026-08-06-paired-retrain-measures-no-gain.md` — the pairing rule
  and metric this reuses
* `docs/changelog/2026-08-06-data-acquisition-ranking.md` — Tier-2 item 4
* `docs/changelog/2026-08-06-market-relative-labels-refuted.md` — the prior case of a
  CV win that the production path refused to confirm
