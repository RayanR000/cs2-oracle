# The Accuracy Series Rests on Two Market Days (2026-08-03)

Investigating the apparent 30d directional inversion — the model predicting
"down" 82% of the time into a cohort that rose 64% — dissolved the inversion and
found something larger underneath. The whole reported accuracy series spans
**one or two distinct forecast dates**, while carrying five-figure sample counts.

## The 30d inversion is not real

The entire 30d ≥$1 cohort is a single forecast date: **2025-12-01 → 2025-12-31**.
All 990 forecasts, one day, one month. Controlling for date, 30d is unremarkable:

| forecast date | 3d | 7d | 14d | 30d |
|---|---|---|---|---|
| 2025-12-01 | 39.1% | 33.4% | 38.7% | **38.7%** |

30d sits between 3d and 14d. It looked catastrophic only because it is the one
horizon whose cohort lands *entirely* on that date, while 3/7/14d also contain
2026-07-17, which lifts their pooled averages. **There is no 30d-specific defect.**

Ruled out along the way: `direction_predicted` agrees with `sign(mid − current_price)`
at 92% on 30d, and the model's own `predicted_price_mid` is below `base_price` on
78% of rows. The label is not inverted relative to the model; the model genuinely
forecasts down. So this was never a sign or label bug.

## The real finding: effective sample size is 2, not 60,737

| horizon / model | n | distinct forecast dates |
|---|---|---|
| 3d lgbm-v3 | 11,009 | **2** |
| 7d lgbm-v3 | 11,059 | **2** |
| 14d lgbm-v3 | 11,040 | **2** |
| 30d lgbm-v3 | 5,461 | **1** |
| 3d/7d global-only, regime | 5,542 ea. | **1** |

The two dates ran in opposite directions, and the model's accuracy tracks the
market rather than the horizon:

| date | market | model (7d) | always-down | always-up | model says "down" |
|---|---|---|---|---|---|
| 2025-12-01 | rising | 33.4% | 29.4% | **62.0%** | 87.2% |
| 2026-07-17 | falling | 63.7% | **76.9%** | 18.6% | 81.2% |

The model predicts "down" 57–87% of the time *regardless of date*. It therefore
scores well when the market falls and badly when it rises. Most of the
horizon-to-horizon variation in the 2026-08-01 report is really *which of these
two dates each cohort happened to contain*.

Every item sharing a forecast date is exposed to the same market-wide move, so
11,000 rows on two dates are far closer to two observations than to 11,000.

### Why the confidence intervals were wrong

`bootstrap_ci` resamples individual records, which assumes independence. Under
clustering it measures only within-day spread, so it reported a tight interval
around a quantity whose real uncertainty is entirely between-day. The stored CIs
are not conservative — they are the wrong quantity.

## Changes

**`distinct_forecast_dates` and `date_coverage_sufficient`** are now in every
`score_cohort` result, and `MIN_FORECAST_DATES = 20` gates the headline. Below
it, `_score_groups` logs a `WARNING` naming the date count and refuses to quote
the figure, rather than printing a percentage the cohort cannot support. The
value is still stored — it is data — but it is no longer *reported* as a
headline. 20 is a judgement call, documented as such: enough to span more than
one market swing, and deliberately well above the 2 on hand.

**`block_bootstrap_ci`** resamples whole forecast dates with replacement and is
reported as `directional_accuracy_ci_clustered_{lower,upper}`. The item-resampled
pair is retained for continuity with the stored series, marked in-code as
understating the uncertainty. Fewer than 2 clusters returns `(None, None)` — one
date carries no information about between-date variation, and any interval from
it would be fabricated.

Against the real stored cohorts:

| horizon | n | dates | DirAcc | naive CI | clustered CI |
|---|---|---|---|---|---|
| 3d | 2,092 | 2 | 45.12% | 43.1–47.3 | 39.1–50.6 |
| 7d | 2,095 | 2 | 49.21% | 47.0–51.3 | **33.4–63.7** |
| 14d | 2,093 | 2 | 45.25% | 43.1–47.3 | 38.7–51.2 |
| 30d | 990 | 1 | 38.69% | 35.8–41.8 | **n/a (<2 dates)** |

The 7d interval widens from 4.3pp to 30pp. Every cohort now reports
`date_coverage_sufficient = False`.

`forecast_date` is carried into the scoring records (added to the `SELECT` in
`_score_frozen_outcomes`); records predating the field report 0 distinct dates
and fail the gate, which is the correct reading of "we cannot tell".

## Why there are only two dates

The forecast pipeline was dead in CI from 2026-07-14 to 07-31
(`2026-07-31-accuracy-work-closed.md`), and backtest maturity is bounded by
archive coverage. The only batches that both exist and have matured are the
2025-12-01 seed batch and the 2026-07-17 run.

**This is not fixable in code.** Accumulating dates is calendar time, and it
requires the daily forecast run to actually be writing. Until then the accuracy
series cannot distinguish model skill from two days of market direction.

## What this retracts

The comparison in `2026-08-03-headline-persisted-and-tier-mirror.md` — "a
constant beats the model at every horizon by 5–26pp" — **pooled these two
opposite regimes**, which made "always down" look like a stable winner. It is
not: the winning constant flips between the dates, and which one wins is only
knowable in hindsight.

A constant does still beat the model on each date taken alone (52–64% vs 33–39%
on 12-01; 73–77% vs 50–64% on 07-17). But with two dates that is a suggestive
pattern, not a measured result, and it should not be quoted as one.

The model's persistent down-bias is real and visible in `direction_predicted`.
Whether it is *costly* is exactly what two dates cannot answer.

## Tests

`tests/test_accuracy_date_clustering.py`, 8 new:

- `test_distinct_forecast_dates_is_in_the_metrics`
- `test_a_single_date_cohort_is_flagged_insufficient` — 5,000 forecasts on one
  day must not read as a sufficient cohort
- `test_sufficiency_threshold_is_met_at_the_minimum`
- `test_minimum_is_greater_than_the_two_dates_currently_stored`
- `test_clustered_ci_is_wider_than_the_naive_one` — two all-or-nothing dates
  must produce an interval reaching both extremes
- `test_ci_is_deterministic`
- `test_single_cluster_yields_no_interval`
- `test_score_cohort_reports_the_clustered_interval`

Full suite: 329 pass.

## Files changed

- `backend/backtest/scoring.py` — `MIN_FORECAST_DATES`, `block_bootstrap_ci`,
  two new metrics
- `backend/scripts/backtest_accuracy.py` — `forecast_date` in the scoring
  records; headline refuses below the gate
- `backend/tests/test_accuracy_date_clustering.py` — new

## Related

- `docs/changelog/2026-08-03-headline-persisted-and-tier-mirror.md` — the
  headline correction this partly retracts
- `docs/changelog/2026-08-01-deterministic-backtest.md` — the series in question
- `docs/changelog/2026-07-31-accuracy-work-closed.md` — the CI outage that left
  only two matured batches
