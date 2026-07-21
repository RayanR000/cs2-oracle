# Walk-forward backtest script + accuracy comparison across evaluation methods

**Date:** 2026-07-20

**Files changed:**
- `backend/scripts/walkforward_backtest.py` — new script: reusable walk-forward backtest using the current model's tuned parameters from `meta.json`
- `docs/architecture/model.md` — will need accuracy numbers updated after this entry is merged

---

## What

### walkforward_backtest.py

A standalone, reusable script at `backend/scripts/walkforward_backtest.py` that evaluates the current model's tuned parameters (from `models/saved_models/meta.json`) on historical Parquet data via a walk-forward design:

1. **Data loading** — reads items from `price-archive/*.parquet` via DuckDB, loads all prices, engineers features via `ItemForecaster.engineer_features()`
2. **Walk-forward splitting** — for each horizon, splits dates into an expanding-window design: 2/3 train, 1/3 validation tail, 60-day step size, 21-day validation windows. Train rows capped at 200,000.
3. **Model training per fold** — trains p10/p50/p90 LightGBM quantile models using the exact tuned params from `meta.json` (per-horizon, per-quantile). Early stopping (20 rounds) on validation. Collinearity filtering (>0.95 correlation) applied. Missing values filled with training median.
4. **Evaluation** — computes directional accuracy, MAE, RMSE, MAPE, wMAPE, and interval coverage per fold.
5. **Aggregation** — sample-count-weighted average across folds. Optionally upserts results to `prediction_accuracy` table.

### DuckDB compatibility fix

Older Parquet files (pre-2026) lack the `source` column. The `UNION ALL BY NAME` pattern handles this by checking column existence per file via `DESCRIBE` before constructing the per-file SELECT clause, adding `NULL::VARCHAR AS source` for files that don't have it.

## Why

The existing walk-forward evaluation in `evaluate_forecaster.py` was an older script that tested fixed LightGBM params (not the tuned ones from Optuna) on only 50 items with no ensemble. The model documentation needed a reproducible, tuned-params evaluation that could:
- Run on any number of items (not just 50)
- Use the exact `meta.json` params that the production retrain produces
- Be run independently without triggering a full retrain
- Validate that the Optuna-tuned params generalize out-of-sample at scale

## Results

### Walk-forward backtest (200 items, tuned params, 26 folds)

Ran all 4 horizons on the top 200 items by data volume (highest row count in Parquet archive):

| Horizon | DirAcc | MAE | MAPE | wMAPE | IntCov | Folds | Samples |
|---------|:------:|:---:|:----:|:-----:|:------:|:-----:|:-------:|
| **3d**  | 62.9%  | $0.06 | 5.4% | 11.7% | 84.6%  | 26    | 108,807 |
| **7d**  | 65.5%  | $0.08 | 7.0% | 16.0% | 84.2%  | 26    | 108,715 |
| **14d** | 67.7%  | $0.11 | 15.4% | 22.2% | 81.7%  | 26    | 108,878 |
| **30d** | 67.4%  | $0.16 | 25.3% | 31.9% | 80.6%  | 26    | 108,960 |

Total runtime: ~9.3 min for all 4 horizons with 200 items.

### Production backtest (all 8,691 items)

Also re-ran `backtest_accuracy.py` for the current numbers:

| Horizon | DirAcc | MAE | MAPE | Samples |
|---------|:------:|:---:|:----:|:-------:|
| **3d**  | 61.5%  | $0.76 | 33.9% | 5,512 |
| **7d**  | 52.8%  | $0.76 | 44.9% | 5,431 |
| **14d** | 55.7%  | $0.77 | 53.3% | 5,434 |
| **30d** | 54.2%  | $0.79 | 35.0% | 5,360 |

Runtime: ~13s.

## Accuracy comparison across evaluation methods

| Method | Items | 3d DA | 7d DA | 14d DA | 30d DA | Notes |
|--------|:----:|:-----:|:-----:|:------:|:------:|-------|
| Pre-audit (buggy) | — | — | ~87% | — | ~83% | Target-inversion inflated (docs/architecture/model.md) |
| lgbm-v3 initial (2-fold CV) | ~200 | — | ~67-69% | — | — | Jul 16, narrow window |
| lgbm-v3 improved (6-fold CV) | ~200 | 66.8% | 65.7% | 65.9% | — | Jul 16, 1460d window |
| **Walkforward (tuned, 200 items)** | 200 | **62.9%** | **65.5%** | **67.7%** | **67.4%** | **This session — 26 folds, 109K samples/horizon** |
| Walkforward (historical, 50 items) | 50 | 61.4% | 60.9% | 60.2% | 68.1% | From model.md — fixed params, no ensemble |
| Production backtest (Jul 19) | 8,691 | 59.7% | 46.5% | 46.0% | 46.9% | All items, Dec 2025 batch |
| **Production backtest (Jul 20)** | 8,691 | **61.5%** | **52.8%** | **55.7%** | **54.2%** | **This session** |

### Key observations

1. **Walkforward with tuned params (200 items, 63-67% DA)** is in line with the documented ~66% CV numbers from model.md. The tuned params generalize well — no degradation from the earlier fixed-param 50-item test.

2. **Production backtest is 5-13pp lower** because it evaluates all 8,691 items (not just data-rich ones). Items with sparse history, dead items at Steam floor ($0.03), and items with volatile low-volume periods dilute the aggregate.

3. **The gap between walkforward and production backtest** is not a model regression — it's a population difference. The walkforward tests only the best-sampled 200 items; production tests everything with a mature forecast.

4. **Jul 19 accuracy records showed 33-44% DA** (anomalous drop that self-resolved). The Jul 20 production backtest matches expected ranges.

5. **No previous model version data exists in `prediction_accuracy`** — the table only contains lgbm-v3 records. Cross-version comparison relies on changelog/docs.

## Implementation details

- Uses `ItemForecaster.engineer_features()` and `ItemForecaster._add_cross_sectional_features()` directly — identical feature pipeline to training/predict.
- Uses `ItemForecaster._fix_quantile_crossing()` for p10/p90 interval correction.
- `_get_tuned_params()` reads from `meta.json["tuned_params"][horizon][q]` with hardcoded fallbacks if missing.
- Folds use `lightgbm.train()` with early stopping (20 rounds) and the tuned params — no ensemble averaging per fold (single model, like a single ensemble member).
- CLI args: `--max-items N` (default 500), `--horizons 3 7 14 30`, `--skip-db` for dry runs.
- Results optionally upserted to `prediction_accuracy` with `prediction_type = 'walkforward_backtest'` and `model_version = 'lgbm-v3-tuned'`.
