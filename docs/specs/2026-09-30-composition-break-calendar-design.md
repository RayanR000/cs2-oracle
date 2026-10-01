# Composition-break calendar

**Date:** 2026-09-30
**Status:** draft, awaiting review
**Source:** deep review §1 / §12.8 (`research/2026-08-19-deep-model-review.md`), next-steps §6
(`research/2026-09-28-next-steps.md`)
**Ships in:** the Mon 2026-10-05 `mode=full` retrain PR, together with next-steps item 16
(remove the `SHRINK_K_GBM` / vol-rank flag plumbing).

## Problem

The consensus price is a vote across whichever sources reported that day. When the source set
changes for the whole cross-section at once, every item's "price" switches basis overnight. A
label or a lookback feature that spans that day measures the switch, not the market.

Production already voids **labels** that span a "collector cutover", which
`ItemForecaster._collection_shift_dates` detects as a >20% day-over-day change in the number of
items. That rule has two gaps, both measured on 2026-09-30 on the durable archive (probe in the
session scratchpad, not committed; reproduction below).

### Gap 1: a source switch with a flat item count (2026-04-16)

On production's train universe (pre-2026, non-iflow, 5,536 items rebuilt from the archive):

| | 2026-04-15 | 2026-04-16 |
|---|---|---|
| Dominant source set | buff163 + csfloat + youpin (4,856 items) | steam_17mafo only (4,888 items) |
| Items present | 5,531 | 5,532 (+0.02%) |
| Items whose source set changed vs the day before | | **100%** |
| Cross-sectional median daily return (≥$1) | | **−1.92%** (sd 0.33%, about 6σ) |

The item-count rule cannot fire here. The 2026-09-28 retrain logged only
`2026-03-22, 07-09, 07-10, 07-11, 07-12` as cutovers, so every label spanning 04-16 trains on a
basis change. At h=30 that is roughly 30 anchor days × 5.5k items.

A composition rule separates break days cleanly. Here "changed" means the share of items present
on both days whose set of voting sources differs:

| Days | Share changed |
|---|---|
| 03-22, 04-16, 07-09, 07-10, 07-11 | 0.9989–1.0000 |
| Highest ordinary day, train universe | 0.56 (Jan–Feb sit around 0.50 because `aggregator_sync` steps in and out) |
| Highest ordinary day, full archive | 0.48 |

### Gap 2: lookback features are never checked

The served model uses 28 features, all `price_technicals` (retrain log, 2026-09-28). Their
lookbacks run up to 200 days (`return_180d`, `price_dist_ma200`). Nothing voids a feature whose
window spans a break. Of the 2026 rows priced ≥$1 on the full archive, these shares have a
lookback crossing a known break:

| Lookback | Share of rows |
|---|---|
| 7-day | 12.4% |
| 14-day | 23.2% |
| 30-day | 44.5% |

This also affects **serving today**. On 2026-09-30, `return_90d`, `return_120d`, `return_180d`,
`price_dist_ma100` and `price_dist_ma200` all reach back across 07-09..07-12, so the served
forecasts still read the July switch as a price move. They will until about January 2027.

### Out of scope: per-item source churn

Item-days whose own source set changed move far more than item-days whose set held. In September
the median |daily return| was 5.9% against 0.08%, and the 90th percentile 49% against 3.4%. This
is per-item and ongoing, so a calendar cannot express it. The cause is also unsettled: illiquid
items naturally both churn sources and move more. It is recorded in the changelog as an open
question; no rule here.

## Design

### Part A: composition breaks join the label span rule (ships ON)

1. **The vote emits the source set.** `_multi_source_voting_sql` adds one column,
   `source_set_hash BIGINT`: `hash(string_agg(DISTINCT COALESCE(source, '<null>'), '|' ORDER BY …))`
   over the `kept` rows, the same rows `n_ask_sources` counts. Only equality is ever compared, so a
   hash is enough and needs no source registry to maintain. The pandas reference
   `_apply_multi_source_voting` gets the same column, and `tests/test_sql_voting.py` asserts the two
   agree, as it already does for every other column. `VOTED_CACHE_VERSION` goes to 12.
2. **Detector.** `ItemForecaster._composition_break_dates(voted) -> frozenset[date]`. Day `d` is a
   break when at least `MIN_DEGENERATE_CROSS_SECTION` (25) items are present on both `d−1` and `d`,
   and the share of them whose `source_set_hash` differs is ≥
   `COMPOSITION_BREAK_FRACTION = 0.90`. Like the item-count rule, it is computed from the universe
   and never from prices, so a real crash cannot trigger it. Pre-2026 rows all hash `<null>`, so
   they never fire.
3. **Calendar.** `break_dates = _collection_shift_dates ∪ _composition_break_dates`. It is
   computed in `fetch_price_history` on the voted frame, before `engineer_features` drops the
   column, and stored on the instance (`self.break_dates`). `prepare_targets` uses
   `self.break_dates` in place of its own `_collection_shift_dates(df)` call. If the attribute is
   unset (a harness that builds frames without `fetch_price_history`), it falls back to the
   item-count rule and logs a WARNING naming the gap. The span rule itself is unchanged:
   `anchor < b <= anchor + h`.
4. **Record.** `label_voiding` (and so `meta.json`) gains `composition_break_dates` next to
   `collection_shift_dates`, plus the union. The "Label voiding" log line names both.

Expected effect: 04-16 joins the five dates already voided. 07-14/07-15 fire the item-count rule
on the full universe but not on the train universe, and the composition rule doesn't change that.
No feature, band or serving code changes in Part A.

### Part B: break-aware lookbacks (ships built, flag OFF)

1. **Flag** `BREAK_AWARE_LOOKBACKS=1`, default off, read in both training and predict. It is
   recorded in `meta.json` as `break_aware_lookbacks`, and a missing key reads as off, so older
   artifacts load unchanged.
2. **Window table.** `FEATURE_LOOKBACK_DAYS: dict[str, int]` declares each served feature's
   calendar lookback, e.g. `return_30d: 30`, `price_dist_ma200: 200`, `rsi_14: 14`,
   `bb_pct_b`/`bb_width: 20` (20-day Bollinger), `autocorr_1d: 2`, `price_accel_7d: 14`. EWM features get a declared effective window of `3 × span`: MACD gets
   78 (span 26), and the signal-line span 9 is inside that. `price_tier: 0`. A test asserts that
   every feature the allowlist admits has an entry, so a new feature cannot slip past the rule.
3. **Rule.** When the flag is on, feature `f` on row date `d` is set to NaN if any break `b`
   satisfies `d − W_f < b ≤ d`. LightGBM handles NaN natively (`FEATURE_NATIVE_NAN` has been on
   since #30). Rows are kept. Dropping them would remove 45% of 2026 at the 30-day window.
4. **Parity.** The same rule runs on the predict frame with the same calendar. The calendar is
   computed on the predict fetch (`PREDICT_FETCH_DAYS = 730`) **before** the
   `PREDICT_TAIL_ITEM_DAYS = 240` per-item trim, so it sees every break the longest window
   (200 days) can reach. An assertion checks fetch depth ≥ the longest declared window.

**Why NaN and not splicing.** Back-adjusting the pre-break series by a per-item level ratio
would keep every row. But it invents an adjustment factor from one noisy day on each side, and
September's per-item jumps have a 49% 90th percentile. It is also the same move
`merge_17mafo_gap.py` made, which the deep review faults for hiding a break instead of removing
it. Rejected.

**Known cost.** With the flag on, `return_90d+` and `price_dist_ma100/200` are NaN for every
2026 row after 03-22 and stay NaN at serve time until about January 2027. That is a large
change to what the model sees, which is why Part B is off until measured.

### Part B gate (a separate PR, after this one)

Paired arms, flag off against flag on, through the production trainer (`ab-statistics` rule:
`_boost_rounds(cv=True)`, `_train_ensemble_member`, fold-clustered intervals from
`backtest/paired_mde.py`).

- Run `paired_mde` first and stop if the MDE exceeds about 2pp.
- **Primary metrics:** mean band width at matched coverage, and interval score, at each horizon.
  This follows invariant 4: band quality, never directional accuracy (DA) alone.
- **Secondary:** a served confirmation through `scripts/replay_serving.py`.
- Flip the default only on a SUPPORTED primary at h=14 and h=30, the horizons the long lookbacks
  feed. Record the result in `experiment_log.csv` whatever it is.

### Item 16, in the same PR

Remove `SHRINK_K_GBM` from `price-forecast.yml`, and remove the env reads, the `meta.json`
write/read of `shrink_k_gbm` / `vol_rank_gbm` / `vol_rank_norm`, the `_*_served()` accessors, and
the serve-time vol-rank multiplier. Loading an old artifact that still carries the keys must keep
working: ignore them, don't raise. The helpers the archived replays call
(`scripts/archive/shrink_k_*.py`) move with those scripts or go with them; the plan decides which.
The workspace rule is met because this PR's merge triggers the 10-05 `mode=full` retrain.

## Interactions

- **PID prereg (item 4).** The 10-05 retrain lands inside the 09-07..10-18 window. Weekly
  retrains were always going to happen there, and the prereg's validity check bounds model
  changes. Part A moves labels only on rows spanning 04-16, which is outside the window, so both
  PID arms are recalibrating the same served forecasts. No prereg edit.
- **Champion–challenger (item 5)** and the **feedback factors** read served rows only. Part A
  changes neither, so no interaction beyond the ordinary retrain.
- **Walk-forward and harnesses** that call `prepare_targets` without `fetch_price_history` keep
  the item-count rule and WARN. `walkforward_backtest.py` has its own loader and does not vote, so
  it gets no composition column. That is a known divergence and is listed in its docstring.

## Testing

- `test_sql_voting.py`: `source_set_hash` agrees between the SQL and pandas votes, and is stable
  under source order and duplicates. NULL source hashes as `<null>`.
- `_composition_break_dates`:
  - a synthetic 100% switch fires;
  - a 56% churn day does not;
  - a cross-section under 25 items does not;
  - a crash with an unchanged source set does not.
- `prepare_targets`:
  - a label spanning a composition-only break is voided;
  - the fallback path WARNs;
  - `label_voiding` carries both lists.
- Part B: a feature whose window spans a break is NaN with the flag on and untouched with it
  off; the window table covers the allowlist; predict and train apply the same mask.
- Item 16: an artifact with the old keys still loads and predicts.
- **Reproduce on the real archive** (in the PR description, not a test): the composition detector
  on the train universe returns exactly `03-22, 04-16, 07-09, 07-10, 07-11`.

## Verification after merge

The 10-05 retrain log should show `composition breaks: [..., 2026-04-16, ...]` and per-horizon
voided label counts above the 09-28 run's 48,319 / 52,116 / 58,595 / 72,461. Check also that
`forecast_date` advances, the vote prints `8 item chunks`, and `Wrote ~22.5k forecasts`.
