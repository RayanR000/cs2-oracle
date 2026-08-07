# The ByMykel bundle is wired in behind a flag, and the paired retrain does not confirm it

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

## What would make this answerable

In rough order of cost:

1. **Repeat the paired retrain N times** and read the distribution of paired diffs
   rather than one draw. Given the observed run-to-run swing this is the minimum, and
   at ~2–4 min per arm it is cheap. It does not fix the MDE, only the reproducibility.
2. **Raise the training budget.** `TRAIN_FEATURE_ROWS = 700_000` puts ~646 items in
   the frame instead of ~99, which is the regime the CV result was measured in. It
   was declined on cost (468.7s vs 104.6s) and that trade-off has not changed, but
   this is the one lever that addresses the actual disagreement.
3. **More folds.** The MDE is fold-bound, and the placebo's behaviour at 8–9 folds is
   itself the reason to distrust the current count.

Until one of those runs, this is not evidence for or against the bundle.

## What was deliberately not done

* **The flag was not enabled**, no production retrain was run, and
  `models/saved_models/` was not touched — both arms train into scratch dirs.
* **`MODEL_ARTIFACT_VERSION` was not bumped.** See above.
* **No repeat runs.** Two runs revealed the instability; characterising it properly is
  item 1 above and was not done here.
* **The 700K budget arm was not run.** It is the most informative next step and also
  the expensive one; that is a cost decision, not something this measurement settles.

## Related

* `docs/changelog/2026-08-06-bymykel-metadata-ingest.md` — the ingest and the CV result
* `docs/changelog/2026-08-06-paired-retrain-measures-no-gain.md` — the pairing rule
  and metric this reuses
* `docs/changelog/2026-08-06-data-acquisition-ranking.md` — Tier-2 item 4
* `docs/changelog/2026-08-06-market-relative-labels-refuted.md` — the prior case of a
  CV win that the production path refused to confirm
