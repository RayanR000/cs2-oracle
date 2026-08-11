# The resolver stays put; the prediction leg is what is wrong — and it is wrong today

Written to settle the question arm A left open: should `backtest/price_resolution.py` follow
serving if `SERVE_OUTLIER_GATED_ANCHOR` becomes the default
(`2026-08-11-serving-anchor-freshness-measured.md`)?

**Ruling: no. Do not move the resolver. Move the prediction leg — and it needs moving whether
or not arm A ever ships, because `ActionableDA`'s selected cohort is 98% artifact right now.**

## Why the resolver must not follow serving

`resolve_anchors` puts **both legs of `actual_ret`** — the forecast-date base and the
target-date actual — through one function, one window, one source. That symmetry is the whole
determinism guarantee, and it exists because of a specific incident: before 2026-08-01 the base
leg was the served `current_price` and the actual leg was a raw single-day voted price, and
differencing two estimators let the same 5,512 forecasts score **61.76% one day and 33.74% the
next**. Making the base leg track a serving flag re-opens exactly that door, and it would do it
for a reason (which price the API publishes) that has nothing to do with scoring.

## What is actually incoherent

Scoring derives the prediction leg as `r_hat = (predicted_mid − base_price) / base_price`
(`backtest/actionable.py:100-102`). But `predicted_mid` was *produced* as
`current_price × (1 + r̂)`. Dividing it by a differently-resolved base does not recover `r̂`; it
recovers `r̂` plus the wedge between the two bases.

Measured on the ops mirror, `forecast_outcomes` rows carrying both columns (`current_price` is
the serving-time value, written through as-is; `base_price` is always archive-resolved):

| cohort | n | wedge median | wedge p90 | rows where the wedge alone exceeds the row's actionable threshold |
|---|---|---|---|---|
| ≥ $1, all horizons | 11,480 | 5.70% | 37.82% | 2,136 (18.6%) |
| ≥ $1, h ∈ {14, 30} | 3,083 | **13.74%** | 54.74% | 1,048 (34.0%) |

The two bases disagree on **85.3%** of rows.

## The consequence, in one line

`ActionableDA` selects on `|r_hat| > threshold`, thresholds 7.2–37.5% by tier and venue.

| | n selected | hit rate |
|---|---|---|
| as scored today (`base_price`) | **1,141** | 55.6% |
| on the price the forecast was quoted from (`current_price`) | **21** | 38.1% |

1,120 rows are **gained** by the wedge; **none** are lost to it. So 98.2% of the actionable
cohort is selected by the serving/backtest base divergence rather than by forecast conviction,
and the 55.6% is not a measurement of the model.

It could not have been. The model's predicted |return| is **median 0.95%, p90 9.01%** against a
**median threshold of 23.1%** — so the honest answer to "how much of this is tradeable" is
*almost none of it*, which is what `2026-08-07-next-steps.md` step 3 predicted as the expected
result. The wedge has been supplying the conviction the model does not have.

## The fix, scoped

`r_hat` divides by `current_price` — the price the prediction was quoted from — falling back to
`base_price` when it is NULL (legacy rows). `actual_ret` is untouched and keeps both legs on
`resolve_anchors`. The scoring query already `LEFT JOIN`s `item_forecasts`, and
`forecast_outcomes.current_price` is already frozen on the row, so `--rescore` stays
archive-free.

Two neighbours that are **not** part of this and should not be swept in:

- **`in_interval` is already coherent.** `low ≤ actual ≤ high` compares served dollars to a
  resolved dollar price. Both legs are dollars; there is no basis mismatch. Band coverage is a
  separate open item (F1: the band is calibrated around a mid it is not served around).
- **`pct_error` normalising by `base_price` is an explicit human ruling** recorded at
  `_derive_verdict`. It is a scale, not a selection rule, so it does not manufacture a cohort.

## What this unblocks

Once `r_hat` uses the served base, the resolver never needs to know which price serving quotes
from, and **arm A becomes scoring-neutral** — its default can be decided on its own merits
rather than on a coupling. That is the answer to the plan's open question.

## Shipped, 2026-08-11

`_prediction_base` in `backtest/actionable.py`: `r_hat` divides by `current_price` when it is
positive, else `base_price`. `r_act` untouched. `scripts/backtest_accuracy.py` now SELECTs
`o.current_price` and carries it on the record — without that the fix is inert, which is what
the end-to-end test in `test_backtest_scoring.py` pins: a +1% forecast under a 40% wedge was
being reported as clearing tier 1's 23.1% bar.

**No re-resolve, and no `model_version` bump.** `ActionableDA` is recomputed from frozen
columns on every run, so there is nothing stored per row to migrate, and the model did not
change — bumping its version string would misattribute a scorer fix. The discontinuity lives in
the persisted `prediction_accuracy` series, so the payload describes itself instead:
`actionable_n_served_basis` and `actionable_n_fallback_basis`, following the `purge` /
`embargo_days` precedent. `fold_records` has no served quote and needs none — it builds
`mid = base × (1 + mid_ret)`, so the resolved base *is* the quote there, and every walkforward
number is unchanged by construction.

The `backtest-scoring` rule said `current_price` is "never scored on". That sentence is now
scoped: absolute for the legs of `actual_ret`, and explicitly *not* for the prediction leg.
Left as-is it would have had the next reader revert this.

## Expect the number to collapse

Fixing this drops the actionable cohort from 1,141 to ~21 rows at h ∈ {14, 30} and moves the
hit rate from 55.6% to 38.1% on a cohort far too small to read. **That is the correct result,
not a regression** — but it changes the meaning of a published metric, so it wants the
`lgbm-v4-embargoed` treatment: a disclosed discontinuity rather than a quiet re-score.
