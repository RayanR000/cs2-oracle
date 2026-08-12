# The served `confidence` label is withdrawn from the API

F2 of `docs/research/2026-08-10-next-steps.md`, which offered "calibrate it or stop publishing
it". Shipped 2026-08-12 as the withdrawal. The column, the writer and the backtest metric all
stay — only the published claim goes.

## What the tag was

`models/forecaster.py` tags a forecast `high` when the directional classifier's max class
probability clears `DIRECTION_CONFIDENCE_HIGH = 0.5`, `low` otherwise. That cut was never
validated against outcomes. The per-horizon `confidence_thresholds` that *were* fitted describe
`_compute_confidence`, which only runs on the no-classifier fallback path — so `high_accuracy` in
`meta.json` and `conf_gap_pp` in the backtest have never been measuring the same thing, and have
been read as if they were.

## The measurement, and the one that does not count

The backtest's own `conf_gap_pp` on the 2026-08-12 run (`31557070748`, the first on the fixed
`in_interval` basis) reads **−4.6 / −4.8 / −2.0 / −38.7pp** at 3/7/14/30d on the ≥$1 `lgbm-v3`
cohort. **Do not use that number.** `backtest/scoring.py:416-419` splits `high`/`low` over the
whole pooled record set, so it carries the same market-composition term that
`2026-08-11-the-da-gap-is-the-market-direction-of-five-dates.md` rules out for cohort contrasts on
a short date panel. Its −38.7pp at h=30 is **one item**: the h=30 panel has a single forecast date
and `n_high = 1`.

Recomputed per (horizon, forecast date) against prod Postgres — 19,917 outcomes, ≥$1,
`model_version LIKE 'lgbm-v3%'`, 2026-07-19 excluded as the dead-band serving config — on the
**11 cells carrying `n_high >= 30`**:

| h | date | n_high | acc_high | n_low | acc_low | gap |
|---|---|---|---|---|---|---|
| 3 | 2026-08-04 | 321 | 33.6% | 728 | 35.0% | −1.38 |
| 3 | 2026-08-05 | 629 | 30.4% | 484 | 26.4% | +3.92 |
| 3 | 2026-08-06 | 571 | 32.7% | 544 | 29.2% | +3.52 |
| 3 | 2026-08-07 | 474 | 39.7% | 480 | 41.7% | −2.00 |
| 7 | 2025-12-01 | 286 | 29.0% | 716 | 35.2% | −6.17 |
| 7 | 2026-07-29 | 710 | 45.8% | 322 | 46.9% | −1.12 |
| 7 | 2026-07-31 | 922 | 44.5% | 110 | 52.7% | −8.26 |
| 7 | 2026-08-01 | 994 | 34.9% | 121 | 37.2% | −2.28 |
| 7 | 2026-08-02 | 715 | 34.7% | 400 | 33.0% | +1.69 |
| 7 | 2026-08-04 | 582 | 41.9% | 381 | 37.3% | +4.65 |
| 14 | 2026-07-18 | 964 | 43.6% | 89 | 49.4% | −5.87 |

**Two readings, and the second is the one that decides it.**

1. **The gap is not measurably negative.** It runs −8.26 to +4.65pp with mixed sign, 7 of 11
   negative. "The tag inverts" is *not* supported on adequately-powered cells — the pooled figure
   said so only because the cohort mix moves with the date. The earlier and much stronger result
   quoted in `tests/test_opportunity_selection.py` (nine cells, high 4–31% vs low 25–55%) was
   measured on a 2026-08-03 panel and does not reproduce on the current classifier.
2. **The `high` cohort is right 29.0–45.8% of the time.** Every powered cell is below a coin flip,
   against `CONFIDENCE_TARGET_ACCURACY = 80.0` — a calibration error of 34–51pp. A field labelled
   "high confidence" on calls that land below 50% is a false claim regardless of where the gap
   points, and a consumer who sees it will weight on it.

So the tag is withdrawn as *uninformative and mislabelled*, not as *inverted*.

## What changed

- `api/schemas.py`: `confidence` removed from `PredictionOut` and `TrendAnalysisOut`.
- `api/routes/items.py`: the five construction sites, the two locals, and the dead
  `ItemForecast.confidence` column in `_build_trending`'s subquery (selected, never used).
- `tests/test_confidence_withdrawn.py`: schema guards plus a source-level guard, matching the
  pattern `test_opportunity_selection.py` established for the 2026-08-03 gate removal — these
  endpoints fall back across a Parquet path and a DB path that uses `DISTINCT ON`, which SQLite
  cannot compile, so not every construction is reachable in the suite.

Full suite: 2,014 passed.

## What deliberately did not change

`ItemForecast.confidence` is still written by `scripts/forecast_prices.py` and still joined into
the backtest. Dropping the column would silently zero `conf_gap_pp`, which is the only instrument
that could ever retire this withdrawal. The forecaster's own cut is untouched: it is an input to
the measurement now, not a product claim.

`_compute_confidence` and the `confidence_thresholds` fit also stay. They are on the fallback path
only and cost training time for a value nothing publishes — worth revisiting, but that is a
training-budget question, not this one.

## Republishing bar

Not "the gap turns positive". `high` must land near its stated target on cells with enough
forecast dates to test — which, per the 50-date constraint in the entry cited above, this panel
does not yet have at any horizon. If the tag is republished on a different basis (a raw
probability rather than a binary label), it needs its own key and its own calibration record.
