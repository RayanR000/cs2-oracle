# Tier-1 retrain speedups + LightGBM 4.6 compat fix

**Date:** 2026-07-17

**Files changed:**
- `models/forecaster.py` — HP caching/reuse, `N_ENSEMBLES` 9→6, `max_feature_rows` 700K→400K, `MAX_BIN` 127→63, SKIP_CV guard, `feature_pre_filter=False` for LightGBM 4.6
- `scripts/forecast_prices.py` — conditional retrain (skip if model fresh + no drift), HP-reuse helpers
- `.github/workflows/price-forecast.yml` — `SKIP_CV=1` env for automated runs
- `tests/test_forecaster.py` — updated `test_ensemble_constants` to 6, added `test_tuned_params_roundtrip`

---

## 1 — HP caching & reuse (biggest lever)

**Problem:** Every retrain ran the full 20-trial Optuna search per quantile (60 fits/horizon = 240 total) even though last week's tuned params are near-optimal. The changelog had previously established that 15→8 trials cost ≈0pp, and 8→20 cost only ~0.3–0.5pp.

**Fix:** Persist `per_quantile_params` (the tuned hyperparams per horizon & quantile) into `meta.json` as `tuned_params`. On a retrain, if cached params exist for all 3 quantiles and `FORCE_HP_SEARCH` is not set, skip Optuna entirely and reuse the cached params. Re-run Optuna only on the first-ever train, or when `FORCE_HP_SEARCH=1` is set.

**Expected impact:** ~–60% of total fits (240 → 0 fits on normal runs). Accuracy impact: ≈0pp (reusing known-good params).

**New env vars:** `FORCE_HP_SEARCH=1` — re-run Optuna even if cached params exist.

## 2 — Ensemble size 9 → 6

`N_ENSEMBLES` reduced from 9 to 6, with `ENSEMBLE_SEEDS` and `ENSEMBLE_FEATURE_FRACTIONS` trimmed to match. The 3→9 expansion was calibrated at +1–2pp; 9→6 is estimated at –0.3–0.5pp.

Expected ~–33% of ensemble fits (108 → 72).

## 3 — Pre-feature-engineering subsample budget 700K → 400K

Reduces feature-engineering wall-clock and makes every downstream LightGBM fit ~1.7× faster. Same full 1460-day calendar window preserved (fewer items, same time span). Estimated accuracy impact: –0.3–0.7pp.

## 4 — MAX_BIN 127 → 63

Faster histogram binning with mild regularization. Consistent across `_optuna_search_params`, `train()`, and `_cv_evaluate_horizon` Dataset constructions. Estimated impact: –0–0.2pp (previously 255→127 was ~0pp).

New class constant `MAX_BIN = 63` (replaces hard-coded `127` across 3 sites).

## 5 — CV skip on automated retrains

When `SKIP_CV=1` is set in the environment, the expanding-window CV evaluation (18 fits/horizon) is skipped on automated retrains. Calibration falls back to the single 21-day holdout split (existing fallback path). Default is off (CV runs) so local/dispatch runs still produce honest multi-regime metrics.

CI workflow (`price-forecast.yml`) sets `SKIP_CV=1`.

## 6 — Conditional retrain

The Monday `full` retrain mode now only retrains when:
- No saved models exist, or
- `FORCE_RETRAIN=1` is set, or
- Model age ≥ `RETRAIN_INTERVAL_DAYS` (default 14), or
- Concept drift was detected in any horizon.

Otherwise, Monday runs predict-only using the existing models. `--train-only` and drift-triggered auto-retrains are unaffected (always train).

New env vars:
- `RETRAIN_INTERVAL_DAYS` — age threshold (default `"14"`)
- `FORCE_RETRAIN=1` — always retrain regardless of age/drift

## 7 — LightGBM 4.6 compat: `feature_pre_filter=False`

The cold retrain uncovered a LightGBM ≥ 4.6 requirement: when the Dataset is built without `feature_pre_filter=False`, LightGBM pre-filters features at the default `min_data_in_leaf=20`. Any training call that uses a lower `min_data_in_leaf` (e.g., 5 from Optuna) raises:

```
LightGBMError: Reducing min_data_in_leaf with feature_pre_filter=true may cause
unexpected behaviour. You need to set feature_pre_filter=false to dynamically
change the min_data_in_leaf.
```

Fixed by adding `"feature_pre_filter": False` to all `ds_params` dicts (Optuna, train, CV) and to the `base_params/train` params. Without this fix, the Monday retrain would fail the first time Optuna samples a `min_data_in_leaf` below 20.

## Files changed summary

| File | Changes |
|---|---|
| `models/forecaster.py` | HP reuse logic, `N_ENSEMBLES` 9→6 (+ trimmed seeds/fractions), `MAX_BIN=63`, `max_feature_rows` 400K default, SKIP_CV guard, `feature_pre_filter=False` at all Dataset/train sites, `tuned_params` persistence in `save_models/load_models`, safe `cv_results` init when CV produces 0 folds |
| `scripts/forecast_prices.py` | Conditional retrain, `_model_age_days`, `_drift_detected` helpers, `import os/json` |
| `.github/workflows/price-forecast.yml` | `SKIP_CV: "1"` env |
| `tests/test_forecaster.py` | `test_ensemble_constants` → 6, `test_tuned_params_roundtrip` added |

## ⚠️ Critical column-ordering bug: ALL prior metrics INVALIDATED

The cold retrain's CV metrics (below) and ALL previous model versions (lgbm-v2, lgbm-v3) are affected by a **column-ordering bug in `fetch_price_history`** that swapped `price` and `volume` data. A new changelog (`2026-07-17-column-order-bug.md`) documents the discovery, fix, and impact in detail.

Until a fresh retrain is run with the fix, every accuracy figure is measuring performance on **volume data, not price data**.

## Expected combined speedup (within ~0.6–1.2pp accuracy budget)

| Scenario | Before (estimated) | After (estimated) |
|---|---|---|
| Local cold retrain (no HP cache) | ~35–40 min | ~18–25 min (Optuna still runs) |
| Local warm retrain (HP reused + SKIP_CV) | ~20–28 min | **~2–5 min** |
| CI cold (2 vCPU, no SKIP_CV) | ~90–120 min | ~45–75 min |
| CI warm (HP reused + SKIP_CV) | ~60–90 min | **~5–15 min** |
| Most Mondays (HP reused, SKIP_CV, conditional skip) | ~60–90 min | **~0 min / predict-only** |

## Remaining

- **Retrain from scratch** with all bugfixes to get valid price-based accuracy metrics
- Validate `feature_pre_filter=False` against the historical walk-forward backtest
- Consider reducing `num_boost_round` cap (1000) after new baseline established
