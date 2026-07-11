# Model Architecture Decisions

## Current Architecture

Single `ItemForecaster` containing **18 LightGBM models**:

```
7d horizon: 3 quantiles (p10, p50, p90) × 3 seeds (42, 73, 91)
30d horizon: 3 quantiles (p10, p50, p90) × 3 seeds (42, 73, 91)
```

Predictions are averaged across seeds per quantile. p10/p90 provide the interval, p50 is the point prediction.

## Decisions

| Component | Current | Keep? | Why |
|-----------|---------|-------|-----|
| Ensemble seeds | 3 | ✅ Keep | 3-5 is the sweet spot. More seeds give diminishing variance reduction. |
| Quantiles | 3 (p10/p50/p90) | ✅ Keep | Minimal set for point prediction + confidence interval. Dropping p50 loses predictions; adding more doesn't help much. |
| Horizons | 2 (7d, 30d) | ✅ Keep | Covers short and medium term. Could add 60d later but not urgent. |
| Model family | LightGBM only | ➕ Add CatBoost | Largest remaining lift opportunity. Different algorithm = different error patterns. Ensemble averaging across families typically gives +1-3pp. |

## What Not To Do

- **Do not replace with a fine-tuned LLM.** LLMs are worse at numerical time series, slower, harder to retrain, and need 100-1000x more data. LightGBM is the right tool for tabular forecasting.
- **Do not add more neural forecasting models** (N-BEATS, PatchTST, etc.) unless accuracy plateaus and you're willing to manage GPU training. The complexity jump isn't justified at 87/83% accuracy.

## Hyperparameter Search

Replaced brute-force grid search with **Optuna Bayesian optimization** (Jul 2026).

| Before | After |
|--------|-------|
| Grid search: 81 combos (3⁴) | Optuna TPE: 30 Bayesian trials |
| Searched 4 params (`num_leaves`, `lr`, `lambda_l1`, `lambda_l2`) | Searches **6 params** (+ `max_depth`, `min_data_in_leaf`) |
| Fixed discrete values per param | Continuous ranges with log-uniform sampling for `lr` |
| All 81 trials run to completion | `MedianPruner` kills bad trials early (n_startup=5, n_warmup=10) |
| 1 trial round per param combo | TPE sampler learns which regions are promising and focuses there |

**Why:** Bayesian search with pruning finds equally good or better params in ~1/3 the time. The grid was wasteful — many combos were nearly tied in validation loss, and the winner was noisy run-to-run. Optuna also let us expand to 6 params without increasing search time.

**Files changed:**
- `backend/requirements.txt` — added `optuna>=3.6.0`
- `backend/models/forecaster.py` — replaced `_grid_search_params` → `_optuna_search_params`, updated `train()` merge logic

## Future Consideration (do this after CatBoost)

- **Expanding-window CV** — train on multiple expanding windows, average predictions across folds. Reduces sensitivity to any single validation window. Would replace the current single 21-day holdout.
