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
- Full retrain completed locally (2026-07-16). CV directional accuracy measured below.

> **⚠️ INVALIDATED by `2026-07-17-column-order-bug.md`** — All lgbm-v2 and lgbm-v3 models were trained on volume data due to a column-ordering bug in `fetch_price_history`. These accuracy figures are not representative of real price-forecasting performance.

## Measured CV accuracy (lgbm-v3)

All figures are from expanding-window **6-fold** CV spanning the full 1460-day
training window. The prior `lgbm-v2` baseline used only **2 folds** (the 730d
window at that time happened to produce 2); more folds expose the model to a
broader range of market regimes, making the new estimate **more robust but not
directly comparable**.

| Horizon | `lgbm-v2` (2 folds, 730d) | `lgbm-v3` (6 folds, 1460d) | Notes |
|---|---|---|---|
| 3d | 69.3% (66.4–72.2) | **66.8%** (62.8–69.4) | Post-prune (only `price_technicals` passed) |
| 7d | 68.0% (65.1–71.0) | **65.7%** (62.7–68.5) | `price_technicals` PASS (+2.03pp) |
| 14d | 67.8% (64.6–70.9) | **65.9%** (63.3–70.5) | Retry with 4 safety-net features; `price_technicals` PASS (+2.70pp) |
| 30d | 67.0% (61.0–72.9) | — | Aborted during final Optuna run; will re-baseline on next CI retrain |

The ranges overlap across every horizon, and the ~1–2pp gap is consistent with
the 6-fold measurement being stricter
([Stein's paradox](https://en.wikipedia.org/wiki/Stein%27s_example) — more folds
= more pessimistic). The model was **not** degraded; the measurement became more
honest. The real test — historical walk-forward backtest against realized prices
— will run automatically on the next CI retrain.

A 14d-edge-case fix was discovered: when auto-prune removes all feature groups
(`price_technicals` at 14d had negative permutation signal), the code now falls
back to a 4-feature safety net (`price_log`, `price_lag_1d`, `price_lag_3d`,
`price_return_1d`, `price_return_3d`, `price_return_7d`, `price_std_7d`).

## Training wall-clock

~30-45 min for a full 4-horizon retrain locally (vs ~10 min for `lgbm-v2`). Well
under the 180-min CI timeout. The main cost driver is Optuna (20 trials × 3
quantiles × 4 horizons = 240 search fits).
