# Quick Post-Processing Model Wins (2026-07-16)

**Date:** 2026-07-16
**Model version:** `lgbm-v2` → `lgbm-v3`

Low-risk, training/inference-only improvements to the LightGBM forecaster. No new
feature groups were added, so the work avoids the diminishing-returns trap documented
in `docs/research/accuracy-opportunities.md` (every completed feature group delivered
only 30-50% of its pre-estimate).

## Changes

| # | Change | Location | Why |
|---|---|---|---|
| 1 | Ensemble expanded 3 → 9 members, each with a distinct `feature_fraction` (column subsampling) in addition to a distinct row-bagging seed | `forecaster.py` `N_ENSEMBLES`, `ENSEMBLE_SEEDS`, `ENSEMBLE_FEATURE_FRACTIONS`; ensemble loop in `train()` | Diversifies the ensemble along both axes; calibrated +1-2pp. Seeds/fractions persisted in `meta.json` |
| 2 | Training window 730d → 1460d (`train()` and `predict()`) | `forecaster.py` `train()`, `predict()` | Captures more full market cycles; enables more expanding-window CV folds (was only 2) → better calibration |
| 3 | Subsample budget 500K → 700K; per-horizon final-fit cap 600K → 700K | `build_training_data()`, `train()`, `scripts/forecast_prices.py` | Keeps per-item time-series continuity with the longer window while bounding feature-engineering memory |
| 4 | Forecast blending + directional smoothing: blend current return-space predictions toward the previous day's forecast (weight 0.15) | `forecaster.py` `_fetch_prior_forecasts()`, `_blend_returns_with_prior()`, `predict()` | Reduces daily direction flip-flopping; inference-only, no training risk. No-ops on first run / retrain (no prior) |
| 5 | Optuna trials 8 → 20 | `forecaster.py` `_optuna_search_params()` | Better hyperparams; +0.5-1pp at ~2.5x Optuna cost |
| 6 | `MODEL_VERSION` `lgbm-v2` → `lgbm-v3` | `scripts/forecast_prices.py` | Any architecture/retrain change increments the version per `docs/research/accuracy-opportunities.md` |

## Validation

- `tests/test_forecaster.py` — 51 tests pass (added `TestForecastBlending`: ensemble
  constants, prior-forecast parsing into return space, blend moves prediction toward
  prior, blend no-ops without prior / weight 0).
- Stale `saved_models/*.txt` + `meta.json` (old 3-ensemble `lgbm-v2`) removed so a
  fresh `lgbm-v3` retrain is forced rather than serving mismatched artifacts.
- Re-baseline directional accuracy via `python scripts/run_task.py backtest` and
  compare against the restored baseline (3d 69.3 / 7d 68.0 / 14d 67.8 / 30d 67.0).
  Docs warn to measure against these, not the older broken-pipeline numbers.

## Expected cumulative lift

~+3-6pp directional (ensemble + blending + window + HP trials), all additive and
low-risk. Training wall-clock grows from ~10 min to an estimated ~30-45 min (still
well under the 180-min CI timeout).
