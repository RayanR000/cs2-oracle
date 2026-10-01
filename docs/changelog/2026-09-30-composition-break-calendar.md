# Labels void across source-composition breaks; break-aware lookbacks built, off

2026-09-30. Next-steps §6 (deep review §12.8) and item 16 of `research/2026-09-28-next-steps.md`.
Spec: `specs/2026-09-30-composition-break-calendar-design.md`. Plan:
`plans/2026-09-30-composition-break-calendar.md`. Ships with the Mon 2026-10-05 `mode=full`
retrain.

## The gap

The consensus price is a vote across whichever sources reported that day. When the source set
changes for the whole cross-section at once, every item's price switches basis overnight, and a
label spanning that day measures the switch, not the market. Production voided labels only across
a >20% day-over-day change in item count (`_collection_shift_dates`). On the train universe
(pre-2026, non-iflow, 5,536 items) that missed two breaks:

- **2026-04-16.** Every item moved from buff163 + csfloat + youpin to steam_17mafo alone, with the
  item count flat (5,531 → 5,532, +0.02%). The cross-sectional median daily return at ≥$1 was
  **−1.92%** (sd 0.33%, about 6σ).
- **2026-01-01.** Every row through 2025-12-31 is NULL-sourced (Steam median sale) and none from
  01-01 is. The median paired item moves **−4.15%** (1,029 items ≥$1). The item-count rule
  fires there only on the full universe, and the 09-28 retrain did not log it.

The share of paired items whose voting-source set changed separates break days cleanly:
0.9989–1.0000 on 01-01, 03-22, 04-16 and 07-09..07-11, against 0.56 on the highest ordinary day
in the train universe and 0.48 on the full archive.

## What shipped

**Part A: on.**

- The vote emits each item-day's `source_set` (voted cache v12). SQL and pandas agree.
- `_composition_break_dates`: day `d` is a break when ≥25 items are present on `d−1` and `d` and
  ≥90% of them changed source set. It reads the universe, never prices.
- `fetch_price_history` records `break_dates` (item-count ∪ composition) before
  `engineer_features` drops the column, and `prepare_targets` voids across it. Without a fetch it
  WARNs and uses the item-count rule alone. `meta.json` carries both lists and the union, and
  `replay_serving.py`'s outcome audit applies the same rule.

Reproduced on the real archive (`prices-2025.parquet` + `prices-2026-01..09`, train universe,
`days_back=400`):

| Rule | Break days |
|---|---|
| Composition | 2026-01-01, 03-22, **04-16**, 07-09, 07-10, 07-11 |
| Item count | 2026-03-22, 07-09, 07-10, 07-11, 07-12 (as the 09-28 retrain logged) |

So 01-01 and 04-16 join the voided days. Features, bands and serving do not change in Part A.

**Part B: built, `BREAK_AWARE_LOOKBACKS` off.** With the flag on, a price-technical feature whose
lookback `(d − W, d]` contains a break is NaN in both training and predict.
`models/lookback_windows.py` declares `W` for every feature, and a test fails when
`engineer_features` emits a price-technical column it doesn't list. Serving follows the
artifact's `break_aware_lookbacks`. The cost is large: `return_90d+` and the 100/200-day averages
would be NaN for every 2026 row after 03-22 until about 2027-01. So it stays off until the
paired band-quality A/B in the spec's "Part B gate" reads SUPPORTED at h=14 and h=30. No
`experiment_log.csv` row until then.

**Item 16: removed.** `SHRINK_K_GBM` / `VOLATILITY_RANK_GBM` (refuted 2026-09-10) lose their
env reads, accessors, helpers, the serve-time vol-rank multiplier, their `meta.json` keys and the
workflow key. An artifact that still carries the keys or the booster files loads unchanged, and
`save_models` deletes the old files. `scripts/archive/shrink_k_vol_rank_ab.py` and
`shrink_k_stability.py` are deleted; `matched_width` moved to `anomaly_band_modulator_ab.py`.
This PR's merge triggers the 10-05 retrain, which satisfies the remove-a-flag-with-a-retrain rule.

## Verify on the 10-05 retrain

Each `Label voiding (h=…)` line shows `(composition: ['2026-01-01', …, '2026-04-16', …])`, and
the per-horizon voided label counts come in above the 09-28 run's 48,319 / 52,116 / 58,595 / 72,461. `forecast_date` advances, the
vote prints `8 item chunks`, and about 22.5k forecasts are written.

## Open: per-item source churn

Item-days whose own source set changed move far more than ones whose set held. In September the
median |daily return| was 5.9% against 0.08%, and the 90th percentile 49% against 3.4%. This is
per-item and ongoing, so a calendar can't express it, and the cause is unsettled: illiquid items
both churn sources and move more. No rule yet.
