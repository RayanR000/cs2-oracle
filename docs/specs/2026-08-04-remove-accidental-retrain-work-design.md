# Remove the Accidental Retrain Work from the Daily Forecast Path

Date: 2026-08-04

## Summary

Every non-Monday forecast run performs a full four-horizon retrain that nothing
asked for, because the drift check compares against a threshold the model cannot
reach. Separately, the predict path loads 1460 days of price history to compute
features whose longest lookback is 180 days, then discards all but three rows per
item.

Removing both cuts the daily CI run from a measured **14 minutes** to an estimated
**3–4 minutes**. Neither change touches the model class, the feature set, or the
hyperparameters. This spec is the first of a planned sequence; it deliberately
contains no accuracy work.

## Background

This spec came out of a scoping pass for a larger question — whether a much
smaller model class would retain most of the current edge. That question is real
and is deferred to a later spec. The two defects below were found while
establishing where the time actually goes, and they are independent of it: they
pay off whether or not the model is ever rewritten.

Measurements throughout come from CI run `30864690456` (2026-08-04 daily run,
`mode=predict-only`, `ubuntu-latest`), retrieved via the run's `forecast.log`
artifact. Phase timings:

| Phase | Wall time |
|---|---|
| Load models + drift check | 3s |
| **Drift-triggered retrain (all 4 horizons)** | **465s** |
| ↳ of which `fetch_price_history` | 137s |
| ↳ of which 14d + 30d DART | 401s |
| Predict | 214s |
| **Total forecasting step** | **~835s (13m55s)** |

## Defect 1 — Drift can never be "not detected"

`scripts/forecast_prices.py:192-204`, the `--predict-only` branch:

```python
elif predict_only and has_models:
    drifted_horizons = []
    for h in ItemForecaster.HORIZONS:
        drift_result = forecaster.check_concept_drift(
            horizon=h, sliding_window=7, threshold=60.0
        )
        if drift_result and drift_result.get("drifted"):
            drifted_horizons.append(h)
    if drifted_horizons:
        logger.warning(
            f"Drift detected for horizons {drifted_horizons} — "
            f"triggering auto-retrain before prediction."
        )
        do_train = True
```

`threshold=60.0` is hardcoded here and again in `_drift_detected` at `:64`. The
model's reproducible production DA is **3d=48.4% / 7d=49.4% / 14d=50.8% /
30d=46.7%** (`docs/architecture/model-optimization.md`, re-resolved on prod
2026-08-02). The threshold sits above the model's ceiling, so the branch fires on
every run. From the 2026-08-04 log:

```
Drift check (3d, 7 windows): avg_acc=27.6%, threshold=60.0%
DRIFT DETECTED (3d): accuracy=27.6% below threshold=60.0%
Drift check (7d, 7 windows): avg_acc=28.9%, threshold=60.0%
...
Drift detected for horizons [3, 7, 14, 30] — triggering auto-retrain before prediction.
Saved models found, retraining...
```

**This cannot be fixed by choosing a better threshold.** The quantity being
thresholded is not a model-health measurement. Per
`docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md`, the stored
accuracy series spans **one or two distinct forecast dates** while carrying
five-figure sample counts, and the model's accuracy on those dates tracks the
market direction rather than the model:

| date | market | model (7d) | always-down |
|---|---|---|---|
| 2025-12-01 | rising | 33.4% | 29.4% |
| 2026-07-17 | falling | 63.7% | 76.9% |

A retrain trigger reading this signal is reacting to which way the market moved,
not to model decay. `check_concept_drift` already guards on window count
(`forecaster.py:4358`: `if len(accuracies) < 3: return None`) but window count is
not date coverage — seven sliding windows can all sit on one forecast date.

Three consequences:

1. **465s of wasted work per non-Monday run**, ~55% of the step's wall time.
2. **The weekly-retrain design is fiction.** The workflow branches on `date +%u`
   to pick `full` on Mondays (`price-forecast.yml:75-80`), but every other day
   retrains anyway.
3. **The served forecasts come from the daily retrain, not Monday's model.** The
   daily one reuses cached HP and skips CV (`Reusing cached HP for 3d (Optuna +
   CV skipped)`, `CV skipped (warm retrain — using cached thresholds)`). Whether
   it is better or worse than Monday's is unverified; that it is *different* and
   unintended is not.

Note that the boosters from these daily retrains are not persisted — the workflow
saves the model cache only for `full` / `train-only` modes
(`price-forecast.yml:185`). So Monday's model survives in the Actions cache while
the day's forecasts are generated from a throwaway.

## Defect 2 — The predict path loads 1460 days to keep 3 rows

`forecaster.py:3603`:

```python
price_df = self.fetch_price_history(days_back=1460, backfilled_only=True)
```

The code already documents the waste, at `:3527-3532`:

```python
# Prediction consumes only the last few rows per item (tail(3) for the
# smoothed current price, last() for the feature vector), yet the whole-frame
# path engineers all 1460 days for every item just to slice that tail off.
# At 8,691 items / 6.1M rows that peaks well past a 16GB CI runner and gets
# SIGKILLed — see docs and runs 30226424193 / 30666903525 / 30668690592.
PREDICT_TAIL_ROWS = 3
```

The existing mitigation (`_engineer_features_chunked`) bounds *memory* but
deliberately costs "~2x the feature-engineering CPU" (`:3545`) because it makes
two passes. It does not reduce the volume of history engineered.

### Why a naive calendar truncation is unsafe

The served feature set mixes two window semantics:

**Calendar-date joins** — `forecaster.py:937-946`:

```python
LAGS = [1, 3, 7, 14, 30, 60, 90, 120, 180]
df["_date_dt"] = pd.to_datetime(df["date"])
for lag in LAGS:
    past = df[["item_id", "_date_dt", "price"]].rename(...)
    past["_date_dt"] = past["_date_dt"] + pd.Timedelta(days=lag)
```

These need **≥183 calendar days** of window (180 + `PREDICT_TAIL_ROWS`).

**Positional row-count rollings** — `:962` and `:1118-1122`:

```python
for window in [7, 14, 20, 30, 60]:
    roll = grouped["price"].rolling(window, min_periods=1)
...
for window in [100, 200]:
    ma = grouped["price"].rolling(window, min_periods=30).mean()
```

These need **≥203 rows** per item (200 + `PREDICT_TAIL_ROWS`).

The archive is roughly 48% dense — 6,155,446 voted rows across 8,691 items over
1460 days is ~708 rows per item, not 1460. A 240-*day* cutoff could therefore
yield ~115 rows and silently change every rolling feature at the serving edge,
which is a train/serve skew of exactly the kind this codebase has been burned by.

### The invariant that makes it safe

Multi-source voting collapses the frame to **one row per item-day**
(`_apply_multi_source_voting`; 8,103,167 raw rows → 6,155,446 voted rows). Rows
are therefore never denser than daily, so **the last N rows always span ≥N
calendar days.** Truncating by rows-per-item satisfies both constraints with one
parameter.

`PREDICT_TAIL_ITEM_DAYS = 240` gives 240 ≥ 203 positional rows and ≥240 ≥ 183
calendar days, with margin.

Note that `price_dist_ma200` is not in the currently served 44-feature set — it is
removed by correlation pruning (`Pruned 39 features (corr>0.95): 132 remaining`).
The 200-row requirement is retained anyway, because pruning runs per-retrain and
could keep `ma200` over `ma100` on any future fit. Sizing to the computed windows
rather than the surviving ones keeps this change decoupled from retrain outcomes.

## Design

### Change 1 — Separate drift alerting from drift-triggered retraining

Drift *alerting* is worth keeping: `AccuracyAlert` rows are a real operational
signal. Drift-*triggered retraining* on a two-date sample is not.

| Component | Change |
|---|---|
| `ItemForecaster.check_concept_drift` | Keep the `AccuracyAlert` write. Add a **date-coverage guard**: return `None` (insufficient evidence) unless the contributing accuracy rows report sufficient forecast-date coverage. |
| `forecast_prices.py:192-204` (predict-only) | Remove the auto-retrain trigger. Log the drift result at `WARNING` without setting `do_train`. |
| `forecast_prices.py:180` (full mode) | Drop `drifted` from the retrain condition; retain `age is None or age >= retrain_interval`. Monday's `full` run already retrains on age. |
| `_drift_detected` (`:59-70`) | Delete. Dropping `drifted` from the `:180` condition removes its only caller. |
| Escape hatch | `ALLOW_DRIFT_RETRAIN=1` restores the previous behaviour on both branches. |

**No new constant is needed.** `backend/backtest/scoring.py` already carries this
machinery, added by the deterministic-backtest work:

```python
MIN_FORECAST_DATES = 20                                    # scoring.py:38
...
"distinct_forecast_dates": distinct_dates,                  # scoring.py:201
"date_coverage_sufficient": distinct_dates >= MIN_FORECAST_DATES,
```

Every `prediction_accuracy.metrics` row written since then already reports whether
its own date coverage is sufficient, alongside a forecast-date-clustered bootstrap
CI (`directional_accuracy_ci_clustered_lower/upper`, `scoring.py:199-200`). The
guard therefore reads `metrics["date_coverage_sufficient"]` rather than
recomputing anything, keeping one source of truth for what "enough dates" means.

The guard **fails closed**: rows predating the field are treated as insufficient.
This matches the intent already recorded at `scoring.py:170-172` — such records
"report 0 distinct dates and fail the sufficiency check, which is the correct
reading of 'we cannot tell'."

The 60.0 threshold stops gating anything once the retrain triggers are removed, so
its unreachability no longer causes harm. It should still be moved out of the two
call sites into a named class constant, carrying a comment that cites the measured
DA of 46.7–50.8%, so the next reader does not have to rediscover that 60% was
never attainable.

### Change 2 — Truncate the predict path by rows per item

- Add `ItemForecaster.PREDICT_TAIL_ITEM_DAYS = 240`, documented against the 203
  positional / 183 calendar requirements above.
- In `predict()`, tail each item to its last `PREDICT_TAIL_ITEM_DAYS` item-days
  **after** voting and before feature engineering.
- Optionally also push a calendar prefilter into the DuckDB fetch to shrink the
  scan and the voting cost.

**The two halves carry different risk, and should be treated separately.**

*The post-voting tail is exactly safe for every item.* Items with more than 240
rows retain full coverage of the 200-row positional and 180-day calendar windows.
Items with 240 rows or fewer are **untouched** — tailing to the last 240 of 100
rows is a no-op. So no item's features change, regardless of history length. This
half needs no coverage measurement.

*The calendar prefilter is where the risk lives*, because it removes rows an item
genuinely has. `PREDICT_MIN_HISTORY_DAYS = 14` (`forecaster.py:153`), so items are
eligible to be forecast with as little as 14 days of history, and many eligible
items will hold well under 240 rows — a prefilter tight enough to matter will
truncate some of them. Rule: adopt a prefilter only if a measured pass shows
≥99.9% of eligible items retain `min(full_row_count, 240)` rows under it. Candidate
starting value 730 days. If that bar is not met, **ship the tail alone** and leave
the fetch at 1460 days. The tail is the larger of the two effects (6.1M → ~2M
engineered rows, across two chunked passes); the prefilter only shortens the 137s
fetch.

## Testing

| Test | Asserts |
|---|---|
| `test_predict_only_never_trains` | `--predict-only` completes without calling `forecaster.train`, given drift-triggering stored accuracy. |
| `test_drift_alert_still_written` | An `AccuracyAlert` row is still created when accuracy is below threshold and date coverage is sufficient. |
| `test_drift_date_coverage_guard` | `check_concept_drift` returns `None` when the stored rows report `date_coverage_sufficient: False`, even with enough sliding windows — and also when the key is absent entirely (fail-closed). |
| `test_allow_drift_retrain_env` | `ALLOW_DRIFT_RETRAIN=1` restores retrain-on-drift. |
| `test_predict_tail_feature_equality` | For a sample of items, the 44 served feature values from the truncated path equal the 1460-day path. **This is the load-bearing test for Change 2.** |
| `test_voted_frame_one_row_per_item_day` | Post-voting frames contain no duplicate `(item_id, date)` pairs — the invariant the row-based truncation rests on. |

Per `AGENTS.md`, `pytest` and `python3 -m py_compile` must pass. No frontend
change is involved, so `npm` checks do not apply.

## Verification

Both changes are verified by their tests plus one CI run:

1. `pytest backend/tests/ -q`
2. Trigger `price-forecast.yml` via `workflow_dispatch` with `mode=predict-only`
3. Confirm from the log: no `TRAINING LIGHTGBM FORECASTER` block, forecasts still
   written for ~8,691 items, and `Verify forecasts were persisted` passes
4. Compare the `Run ML price forecasting` step duration against the 835s baseline

Accuracy is deliberately **not** re-measured for this spec. Change 1 removes an
unintended retrain; Change 2 is covered by an exact-equality test. Neither alters
the model class, feature set, or hyperparameters, so there is no accuracy
hypothesis to test. Re-running `walkforward_backtest.py` here would spend 60–90
minutes to produce a number whose confidence interval could not resolve a change
this design asserts is zero.

## Expected outcome

| | Now (measured) | After |
|---|---|---|
| Daily CI forecast step | 835s | ~180–240s (est.) |
| ↳ drift-triggered retrain | 465s | 0s (measured removal) |
| ↳ `fetch_price_history` | 137s | ~70s (est., **only if the calendar prefilter clears its coverage bar**; otherwise 137s) |
| ↳ predict | 214s | ~90–120s (est.) |
| Local deliberate retrain | 12–16 min | unchanged |

Only the 465s removal is a measured figure. The predict-path estimates scale from
the row-count reduction (6.1M → ~2M engineered rows) and are not measured until
implementation.

**This spec does little for local retrain iteration time.** Training legitimately
needs the full 1460-day window, so Change 2 does not apply to it, and Change 1
only stops unrequested retrains rather than speeding up requested ones. Local
iteration speed is addressed by the deferred minimal-model spec, where the 462s of
ensemble training is the target.

## Out of scope

- **Skipping computation of non-allowlisted features.** `FEATURE_GROUP_ALLOWLIST
  = ["price_technicals"]` reduces 132 features to 44, but correlation pruning runs
  *before* the allowlist is applied (`forecaster.py:2431-2436`), so which features
  survive pruning depends on the discarded columns (`forecaster.py:2431-2436`).
  Skipping their computation would change the served feature set. That is a model
  change.
- **Caching the voted frame in CI.** `fetch_price_history` already writes
  `backend/data/voted_<key>.parquet` in ~1s, but CI runners are ephemeral so it is
  cold every run. Adding it to the Actions cache would save a further ~137s/day.
  Deferred as an infrastructure decision with its own staleness risk — the cache
  key cannot see code changes, which is why `VOTED_CACHE_VERSION` exists.
- **All accuracy work**, including the findings below.

## Findings recorded for later specs

Neither of these is actioned here. Both are accuracy-side and would make training
*slower*, so they belong with the deferred model work.

1. **The model trains on 133 items.** `Stratified subsample: 133/7,879 items,
   95,721/5,891,875 rows (budget 100,000)`. It then serves forecasts for 8,691
   items. `max_feature_rows` defaults to 100,000 and the `train(max_rows=700_000)`
   call at `forecast_prices.py:209` does not propagate to it.
   `docs/retrain-optimization-analysis.md` estimates "~450 items" at this budget;
   the measured figure is 133.

2. **`walkforward_backtest.py` does not use the clustered scorer that already
   exists.** `backend/backtest/scoring.py` computes forecast-date-clustered
   bootstrap CIs and a `date_coverage_sufficient` flag (`:175-202`), but
   `walkforward_backtest.py:316-321` rolls its own aggregation, weighting each
   fold's metrics by `sample_count` — treating correlated item-rows within a fold
   as independent observations. That is the error identified in
   `docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md`. So the
   repository's only fresh-model gate reports inflated precision *despite* the
   correct machinery being one import away. Routing it through `score_cohort` is
   the likely fix and the natural first task of the measurement-rig spec.
