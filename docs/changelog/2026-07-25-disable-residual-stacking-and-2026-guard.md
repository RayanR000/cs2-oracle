# Disable residual stacking, drop stale 2026 exclusion guard

**Date:** 2026-07-25

**Files changed:**
- `models/forecaster.py` — `STACK_RESIDUALS = False`, gated in both `train()` and `predict()`; removed the 2026-row exclusion guard in `build_training_data`
- `tests/test_forecaster.py` — replaced `TestPredictEnsembleSafe::test_distribution_shift_guard_*` with `TestResidualStackingDisabled` and `test_build_training_data_includes_2026`
- `scripts/backtest_accuracy.py` — batched the `forecast_outcomes` delete (see bug below)
- `models/saved_models/` — retrained; pre-fix models backed up to `models/saved_models_pre2026_20260725_182444/`

## What and why

**Residual stacking disabled.** `STACK_RESIDUALS` fit a Ridge regression on LightGBM residuals using *raw, unscaled* feature values, then added its prediction at serving time. Unscaled inputs mean the correction has no bound: a penny item with a legitimately large `return_14d` (e.g. +900%) produced a linear correction in the +100,000%+ range. On the 2026-inclusive retrain this over-corrected 99% of items across every price tier (median +184% for penny items, -480% for mid-price) and inverted quantile ordering on 100% of predictions. The served direction already comes from the directional classifier and interval width from conformal calibration, so the corrector was both redundant and actively dangerous. Gating on `STACK_RESIDUALS` in `predict()` (not just `train()`) ensures previously-saved residual models on disk stop being applied immediately, without needing to delete them.

**2026 exclusion guard removed.** `build_training_data` used to drop all 2026 rows — a temporary patch for the May–June 2026 archive gap, which left 2026 sparse (some single-day, ~352-item slices) and collapsed the 7d validation window to noise. That gap is now backfilled: 2026 is continuous, ~5,360 items/day, zero <50-item days. Keeping the guard would silently throw away good, current-regime training and CV data going forward.

## Bug found while validating: SQLite delete exceeds variable limit

`_store_forecast_outcomes()` in `scripts/backtest_accuracy.py` deleted matched `forecast_outcomes` rows via a single `DELETE ... WHERE forecast_id IN (...)` built from the full `existing_ids` set. SQLite caps bound parameters at 999; once replaced rows exceeded that (32,821 in this run) the delete raised `sqlite3.OperationalError: too many SQL variables`. The lookup query directly above it already batched by 900 — the delete just hadn't been updated to match. Fixed by chunking the delete the same way.

## Validation

Full test suite: 179 passed. Backtest (`DATABASE_URL="sqlite:///./cs2_market.db" python scripts/backtest_accuracy.py`) against the retrained model, compared to the `lgbm-v3` baseline in `docs/next-steps-tier1.md` — all four horizons within the ≤1.5pp accuracy budget:

| Horizon | Baseline | Now | Δ |
|---------|:--------:|:---:|:-:|
| 3d      | 58.15%   | 57.9% | 0.25pp |
| 7d      | 57.41%   | 57.1% | 0.31pp |
| 14d     | 55.08%   | 54.8% | 0.28pp |
| 30d     | 55.15%   | 54.9% | 0.25pp |

No regression from either change.

## Note on absolute accuracy level

These ~55–58% directional-accuracy numbers are not a regression introduced here — they're the same production-backtest metric documented in `docs/architecture/model.md`'s Historical Accuracy Timeline, which has never been near 70% for this metric. The ~70–87% figures elsewhere in that timeline are 3-class walk-forward/CV accuracy on a 200-item, data-rich subset (and one entry, 87%, is flagged there as buggy/inflated from a target-inversion bug). The production backtest evaluates all ~8,691 items — including sparse-history and dead items — against a binary up/down outcome, which is a harder and more representative test; the doc notes walk-forward numbers run 5–13pp higher than production backtest for exactly this reason. 55–58% vs a ~50% coin-flip (and the much lower "baseline" model shown in the backtest's own comparison column) is the correct frame for this metric, not the 70%+ walk-forward figures.
