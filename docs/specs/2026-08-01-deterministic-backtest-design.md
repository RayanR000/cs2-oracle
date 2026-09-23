# Deterministic Forecast Backtest — Design

**Date:** 2026-08-01
**Status:** ✅ **IMPLEMENTED AND CLOSED (2026-08-01).** All 9 tasks of
`docs/plans/2026-08-01-deterministic-backtest.md` landed; the `backend/backtest/`
package, migration `0019_freeze_forecast_outcome_actuals.py` and the determinism test at
`backend/tests/test_backtest_scoring.py:1204` are the evidence. Outcome, including the prod
backfill of 60,737 rewritten outcomes, in `docs/changelog/2026-08-01-deterministic-backtest.md`.
**Scope:** `backend/scripts/backtest_accuracy.py` and the `forecast_outcomes` /
`prediction_accuracy` schemas.

## Problem

The live accuracy metric is not reproducible. The same forecast cohort, re-evaluated
on different days, produces wildly different directional accuracy:

| Evaluation date | 3d DirAcc, `lgbm-v3`, n=5512 |
|---|---|
| 2026-07-17, 07-18 | 61.76% |
| **2026-07-19** | **33.74%** |
| 2026-07-20, 07-22 | 61.54% |
| 2026-07-25 | 57.91% |

An identical sample count across all six evaluations rules out cohort composition.
The evaluator is moving, not the model.

### Root cause: the two legs of `actual_ret` are different estimators

`backtest_accuracy.py:326` computes:

```python
actual_ret = (actual - current) / current
```

- `current` is `item_forecasts.current_price`, written at forecast time as the
  **median of the item's last 3 observed prices** (`forecaster.py:3510-3525`).
  Note the smoothing is unconditional — line 3525 is a `fillna`, so the median is
  always used. The >10% deviation check at 3519-3521 only emits a log line; it
  never changes the value.
- `actual` is the **single-day voted `mean_price`** for the target date, read
  fresh from the archive on every run (`backtest_accuracy.py:180`).
- `FLAT_TOLERANCE` is `0.005` (`backtest_accuracy.py:36`).

Two different estimators differenced against a 0.5% band. Whenever the archive
gains or revises source rows for a target date, every item's `actual_ret` shifts
together and the cohort's direction labels flip en masse.

### Contributing factor: the evaluated universe is dominated by tick noise

3d `lgbm-v3` cohort, by price tier:

| Tier | n | DirAcc | % actual "flat" | $0.01 as % of price |
|---|---|---|---|---|
| <$0.50 | 7,978 | 58.8% | 27.6% | **20.0%** |
| $0.50–1 | 950 | 62.5% | 1.3% | 1.5% |
| $1–5 | 1,408 | 62.1% | 0.7% | 0.5% |
| $5–20 | 476 | 62.2% | 0.8% | 0.1% |
| ≥$20 | 212 | 59.4% | 0.9% | 0.03% |

(This table splits <$1 into two buckets for analysis. The `_price_tier` helper
used by the schema in §4 does not — its tier 0 is the whole <$1 range.)

72% of the evaluated universe is sub-$0.50 items, where one tick is a 20% move.
Their 27.6% flat rate against ~1% elsewhere is the quantization signature: for
those items the up/flat/down label is decided by which tick the voted median
lands on. CI passes no `--min-price`, so all of them are included and they drive
the headline number.

## Non-goals

This does not reopen the prediction-accuracy roadmap closed on 2026-07-31
(`docs/changelog/2026-07-31-accuracy-work-closed.md`). No new features, no model
architecture changes, and no work on the feature A/B harness. This is the *live
serving* backtest, a distinct artifact from the A/B harness whose noise floor
closed that roadmap.

## Design

### 1. Architecture: a pure scorer over frozen inputs

`backtest_forecasts` currently interleaves resolution, scoring, and storage in one
loop, which is why the metric cannot be tested. Split into three stages:

| Stage | Purity | Responsibility |
|---|---|---|
| `resolve_outcomes` | impure — reads archive | For each newly-mature forecast, compute `base_price` and `actual_price`. Insert-only. |
| `score_outcomes` | **pure** | Frozen rows → metrics. No I/O, no archive, no clock. |
| `store_metrics` | impure — writes | Tiered metrics → `prediction_accuracy`. |

The determinism guarantee lives at the boundary: once resolution has run, the
reported number is a pure function of stored data.

### 2. The estimator

One shared helper, called for **both** legs:

```
_smoothed_price(slug, anchor_date) =
    median of the last 3 observed voted daily prices at or before anchor_date
```

- Voting via `ItemForecaster._apply_multi_source_voting` — the same function the
  training pipeline uses.
- Row-based last-3, mirroring `tail(3)` at `forecaster.py:3513`, not a calendar
  window.
- `base` anchors at `forecast_date`; `actual` anchors at `target_date`.
- `actual_ret = (actual - base) / base`.

`item_forecasts.current_price` is no longer read for scoring. Because both legs
are now the same function of the same source, a flat price series yields exactly
`0.0` — which it does not today.

Row-based lookback is dense in practice. Measured over `prices-2026.parquet`
from 2025-12-01, the span of each item-day's 3-observation window:

| Window span | share |
|---|---|
| ≤3 days | 95.81% |
| 4–7 days | 1.28% |
| 8–30 days | 0.30% |
| >30 days | 0.44% |
| fewer than 3 observations | 2.17% |

### 3. Freezing the actuals, not the metrics

`forecast_outcomes` gains `base_price` and `resolved_at`. Resolution becomes
insert-only:

- A `forecast_id` already resolved is skipped; its `base_price` and
  `actual_price` never change.
- `_store_forecast_outcomes` drops its delete-then-insert
  (`backtest_accuracy.py:221-233`).
- Metrics always recompute from the stored `base`/`actual`, so a scoring fix
  re-derives cleanly with no archive access.

Two escape hatches, neither the default:

- `--rescore` — recompute metrics from frozen actuals. Cheap, no archive read.
- `--reresolve` — re-resolve actuals from the archive, for a genuine resolution
  bug.

**One-time backfill.** The 65,642 existing outcomes carry an `actual_price`
computed under the old estimator and no `base_price`. They are re-resolved once
under the new estimator and then frozen. The historical accuracy series will
shift as a result. That is the fix landing, not a regression, and the changelog
must say so explicitly.

### 4. Tiered metrics

`prediction_accuracy` gains a `price_tier` dimension, reusing the existing
`_price_tier` helper (0–4). One row per tier, plus an `all` row. The upsert key
becomes `(prediction_type, evaluation_date, horizon_days, model_version,
price_tier)`.

Tier assignment uses the frozen `base_price`, not `current_price` — otherwise
tiering reintroduces the dependency just removed.

The headline logged figure aggregates tiers 1–4 (≥$1). The log line reports the
headline and the sub-$1 cohort separately, so "the model is worse on penny items"
stays visible rather than being averaged in or filtered away.

### 5. Error handling and gates

- **Staleness cap.** Reject a leg whose 3-observation window spans more than 7
  days. This matches `FALLBACK_MAX_AGE_DAYS=7` from `db5bddb`, keeping one
  staleness convention in the codebase. Costs ~0.74% of item-days.
- **Fewer than 3 observations** (2.17%): resolve on the 1 or 2 that exist,
  provided they fall within the same 7-day span. Both legs use the identical
  rule, so symmetry holds.
- **Missing archive → hard fail.** `_load_actual_prices` currently warns and
  returns `{}` (`backtest_accuracy.py:125-127`), producing a green run with zero
  actuals — the exact shape `324cfff` was written to eliminate.
- **Unresolvable-rate gate.** Fail the run if more than 10% of mature forecasts
  cannot be resolved. Silent shrinkage of the evaluated cohort is how the current
  metric moves unnoticed.

### 6. Testing

Test-driven; the first test is the bug itself.

1. **Determinism under archive revision** — resolve, score, add a new source row
   for an already-resolved target date, re-run, assert identical metrics. Direct
   regression test for the 61.76 → 33.74 → 61.54 swing.
2. **Estimator symmetry** — a synthetic flat price series gives `actual_ret ==
   0.0` exactly. Fails today.
3. **Freeze invariance** — re-running resolution never mutates `base_price` or
   `actual_price`; only `--reresolve` does.
4. **Tier partition** — per-tier sample counts sum to the `all` row, and no
   forecast lands in two tiers.
5. **Staleness cap** — a leg with observations 40 days apart is rejected, on both
   legs symmetrically.
6. **Purity** — `score_outcomes` twice on identical input gives identical output,
   with the archive absent from the filesystem.

Mutation-check the load-bearing tests as `db5bddb` did for its fallback tests:
disable the cap and confirm the test genuinely fails.

## Expected outcome

A live directional-accuracy figure that is stable across re-runs and reflects the
model rather than archive churn. The 07-24 CV work predicted roughly 55%; the
36–42% currently reported is likely measurement error. This design does not
attempt to improve accuracy — it makes accuracy measurable, which gates the two
follow-on items (drift threshold and regime-model persistence, conformal
recalibration).

## Related

- `docs/changelog/2026-08-01-forecast-pipeline-restored-and-collector-guards.md`
- `docs/changelog/2026-07-31-accuracy-work-closed.md`
- `docs/changelog/2026-07-24-directional-classifier-and-honest-serving-audit.md`
