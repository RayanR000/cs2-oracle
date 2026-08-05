# CV/Production Cohort Parity

Date: 2026-08-05
Status: **implemented** 2026-08-05 — `models/forecaster.py`, tests in
`tests/test_cv_cohort_parity.py`. The numbers below are not yet reproduced from
a retrain; the next full training run is what confirms them (see Verification).

## Problem

`cv_results.mean_classifier_acc` scores **all price tiers**. The production
headline scores **≥$1 only** (`HEADLINE_MIN_TIER = 1` in `backtest/scoring.py`).
They are different populations, so every comparison between them — including the
"~20pp train/serve gap" that five separate hypotheses failed to explain — is
partly a cohort artifact rather than a model defect.

The offline frame is **82.6% tier-0**, so the quoted CV figure is approximately
the penny-item score.

## Evidence

Measured 2026-08-05 by running the saved classifiers over
`models/saved_models/engineered_data.parquet` (6.1M rows) with the label rule
actually in use (fixed ±0.5% — the vol-scaled band is dead code, both
`forecaster.py:3022` and `:4556` pass `sigma=None`) and the date-merged forward
return from `scripts/diagnose_direction_prior.py`.

The replication validates against `meta.json`: all-tiers 67.1 / 67.6 / 68.7 /
69.4% vs stored `mean_classifier_acc` 66.1 / 66.6 / 68.4 / 70.8%.

| horizon | offline <$1 | offline ≥$1 | production ≥$1 | residual gap |
|---|---|---|---|---|
| 3d | 70.2% | 52.1% | 45.1% | −7.0pp |
| 7d | 70.4% | 54.2% | 49.2% | −5.0pp |
| 14d | 71.1% | 57.5% | 45.3% | −12.2pp |
| 30d | 71.6% | 59.1% | 38.7% | −20.4pp |

Two facts fall out and should be recorded whatever is built:

- **A real gap survives cohort correction: 5–20pp, widening with horizon.** It is
  smaller than the ~20pp uniform figure previously chased, and its horizon slope
  is a signal the uniform framing hid.
- **On ≥$1 items the classifier predicts `flat` 0.0% of the time**, at every
  horizon. It is effectively a binary up/down model there, so ≥$1 rows whose
  price did not move are always scored wrong. All of its flat predictions
  (39–41%) are penny items.

Carry-forward staleness is tier-0-only: `actual_price` bit-identical to
`base_price` runs 37–42% at tier 0 and **0–1.8% at every tier ≥$1**. The ≥$1
headline is therefore *not* inflated by stale prices — it is simply below 50%.

## Change

In the CV fold loop (`models/forecaster.py:4558-4560`), compute the classifier
accuracy a second time restricted to `val_df["price_tier"] >= 1`, and carry it
through `fold_metrics` → `cv_results` as a **new** key
(`mean_classifier_acc_ge1`), alongside the existing all-tiers key.

**Add, do not replace.** `mean_dir_acc` feeds the edge-vs-baseline trust gate at
`forecaster.py:3183-3210` and the confidence-threshold calibration; repointing the
primary metric would silently move both. Continuity of the stored series also
matters — `mean_classifier_acc` is what every historical `meta.json` holds.

`price_tier` is already in `feature_cols`, so `val_df` carries it at that site;
no new plumbing and no extra model fits. Cost is one extra mean per fold.

## Verification

No retrain is needed to know whether the numbers are right. The offline ≥$1
figures above (52.1 / 54.2 / 57.5 / 59.1%) were measured on the full frame, so a
retrain's per-fold ≥$1 accuracy should land in that neighbourhood — somewhat
lower, since CV folds are out-of-fold and these are in-sample. A result near
67–71% would mean the tier filter did not apply.

Assert in a test that the two keys differ on a frame containing both tiers, and
that the ≥$1 key is absent rather than `0.0` when a fold has no ≥$1 rows — the
same empty-partition rule `score_cohort` follows.

### What was actually built

`tests/test_cv_cohort_parity.py`, six tests, all green (full suite 680 passed):

- fold metrics carry `classifier_accuracy_ge1` and it sits >5pp below the pooled
  figure on a frame whose penny half never moves and whose dollar half always
  does — an unapplied filter makes the two equal;
- `None`, not `0.0`, when a fold is all-penny, and when the frame has no
  `price_tier` column at all (CV is also driven over frames that predate it);
- `cv_results[h]["mean_classifier_acc_ge1"]` equals the mean of the folds that
  had a ≥$1 cohort, driven through the real `train()`;
- **`edge_vs_best_baseline` still equals `mean_classifier_acc − best_baseline`**
  — the regression guard for "add, do not replace";
- the split reuses `HEADLINE_MIN_TIER` imported from `backtest.scoring`, so the
  CV cohort cannot drift from the headline's.

The real frame was checked directly rather than assumed: `engineered_data.parquet`
is 82.6% tier-0 / 17.4% ≥$1 across 6,106,622 rows, and its `price_tier` banding
reproduces exactly from `price`. `cv_results` is serialized wholesale into
`meta.json`, so the new key persists with no extra plumbing.

The CV log line now reads `classifier=X% (>=$1: Y%)`, so the comparable number is
visible in run output instead of only in the artifact.

## Non-goals

- The serving down-bias (`serving transform` skews toward `down`; the 3,149
  low-history items added 2026-07-29 are median-filled on 14 features). Separate
  change, and unverifiable until those cohorts mature.
- Deleting the dead vol-scaled label apparatus. Worth doing, unrelated.
- Changing `HEADLINE_MIN_TIER`, adding features, or any model-class work.
- The `directional_accuracy_ci_*` units bug (point estimate is a percent, the
  clustered CI a fraction, `backtest/scoring.py:185` vs `:199-200`).
