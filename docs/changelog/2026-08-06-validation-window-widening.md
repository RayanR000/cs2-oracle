# Voided labels no longer demote the production split to a positional one

**Date:** 2026-08-06
**Change:** `_train_horizon_inline`'s train/val split now widens its trailing
calendar window when voided labels starve it, instead of falling through to a
positional 80/20 slice. Fixes side effect 1 of
`2026-08-06-paired-retrain-measures-no-gain.md`.

## The defect

`prepare_targets` voids labels built across a fabricated archive day
(`2026-08-06-scale-free-features-and-fabricated-labels.md`) and
`_train_horizon_inline` drops the voided rows. The fabricated days — the
2026-07-16 / 07-22 re-published snapshots and the 2026-07-09/10 collector
cutovers — sit **inside** the trailing `VALIDATION_WINDOW_DAYS`, and the span
rule voids a horizon-wide band around each cutover. So the window lost rows in
proportion to the horizon, and under the 2,000-row floor the split fell through
to `tdf.iloc[:int(len*0.8)]`.

That is not a thinner version of the same thing. Validation stopped being the
recent 30 days and became the last 20% of the date-sorted frame — roughly ten
months. Early stopping, the Optuna objective and the directional classifier's
stopping set all read that window.

At the production default of `TRAIN_FEATURE_ROWS=100_000` it fired at **three of
four horizons**: 3d ran 90,860 train / 22,788 val against control's 112,411 /
2,445. Only the 600K measurement arms escaped it, which is why the paired read
was clean and production was not.

## The fix

`_choose_validation_split` walks the trailing window backwards one date at a
time until it clears `MIN_VAL_ROWS` (2000) and `MIN_VAL_DATES` (7), bounded by
`MAX_VALIDATION_WINDOW_DAYS` (90). Validation stays a recent contiguous calendar
window — just a wider one — which is the graceful degradation the fallback
should have been. The positional split survives only for a frame too small for
widening to rescue, and is still purged.

The whole decision moved into `_build_production_split` (window choice, purge,
row cap, fallback) because `_train_horizon_inline` is the full Optuna + ensemble
path and cannot be driven from a test. The caller is now one line.

The 2,000/7 floors became `MIN_VAL_ROWS` / `MIN_VAL_DATES`; the feature-group
permutation gate already used the same two numbers inline and now reads the
constants.

## Verified at the production budget

Train-only cold run, `TRAIN_FEATURE_ROWS=100000`, `FORECAST_MODEL_DIR` into a
scratch directory (`models/saved_models/` untouched):

| horizon | voided | window | val rows / dates | train rows | before |
|---|---|---|---|---|---|
| 3d | 916 | **34d** | 2,079 / 21 | 111,564 | 1,683 val → positional |
| 7d | 1,444 | **37d** | 2,079 / 21 | 110,107 | 1,386 val → positional |
| 14d | 2,295 | **41d** | 2,079 / 21 | 107,655 | 990 val → positional |
| 30d | 4,231 | 30d (unchanged) | 3,069 | 101,417 | date split already held |

No fallback warning at any horizon. Train rows are back within ~1% of control's
112,411 instead of 19% below it. The voided counts reproduce the earlier entry's
916 / 1,444 / 2,295 / 4,231 exactly, so the label rule itself is unchanged.

The three widened windows all land on 2,079 rows / 21 dates: widening stops at
the first date that crosses the floor, and the frame carries ~99 rows per date.

A side benefit: feature-group validation now runs at all four horizons. Post-void
it was being skipped at 3d/7d/14d by the same floor, which is how the change
could have silently pruned features on top of everything else.

## Tests

`tests/test_validation_window_widening.py`, 27 tests: the window choice
(healthy / row-starved / date-starved / capped / empty), the assembled split
(trailing window, purge, positional fallback still purged, row cap preserving the
calendar window), and an end-to-end pass over a panel carrying both real
fabricated-day shapes at all four horizons. Full suite 867 passed.

## Still open

* **Nothing was retrained in production.** The shipped artifact still carries the
  positional-split hyperparameters and stopping points at 3d/7d/14d.
* **The 600K paired read stands unchanged** — the fixes measure no gain
  (`2026-08-06-paired-retrain-measures-no-gain.md`). This fix does not claim
  accuracy; it removes a defect that made the production configuration differ
  from the one that was measured. Whether it moves the number is unmeasured.
* **`docs/research/2026-08-06-model-review-plain-english.md` is still superseded** on both
  the "improvement is not proven" framing and the 20-minute estimate.
* **Side effect 2 is not addressed:** voiding still costs a CV fold at 7d and
  14d, and shifts 3d's last fold window.
