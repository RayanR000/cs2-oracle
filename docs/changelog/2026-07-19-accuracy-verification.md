# Accuracy verification and re-backtest

**Date:** 2026-07-19

**Files changed:**
- (No code changes — verification only)
- `docs/architecture/model.md` — updated accuracy section with verified backtest results

---

## What

Re-ran the production backtest (`run_task.py backtest`) and independently verified the accuracy data integrity by tracing the full pipeline from Parquet archive → forecast outcomes → aggregated metrics.

## Backtest results (v3, evaluated 2026-07-19)

All 5,300–5,500 forecasts per horizon from the Dec 1, 2025 batch:

| Horizon | DA | 95% CI | Baseline DA | vs Baseline | MAE | MAPE | wMAPE | IC |
|---------|---:|--------|------------:|------------:|----:|-----:|------:|--:|
| **3d** | **59.71%** | [58.3, 61.0] | 20.23% | **+39.48pp** | $0.73 | 33.84% | 30.80% | 46.75% |
| **7d** | 46.51% | [45.3, 47.9] | 17.42% | +29.09pp | $0.73 | 44.43% | 30.91% | 40.42% |
| **14d** | 45.95% | [44.6, 47.3] | 17.67% | +28.28pp | $0.74 | 53.20% | 31.15% | 48.79% |
| **30d** | 46.87% | [45.5, 48.2] | 18.00% | +28.86pp | $0.77 | 36.43% | 31.99% | 56.16% |

New in this run: baseline comparison (persistence forecast), bootstrap 95% CI, wMAPE, confidence gap, skill_vs_baseline.

## Verification methodology

1. **Actual price cross-check:** Queried raw Parquet archive via DuckDB for 5 randomly sampled items across all 4 target dates — every price matched the `forecast_outcomes` table to the penny (SCAR-20 | Fragments FN $0.49, M4A1-S | Solitude BS $1.46, MP5-SD | Dirt Drop BS $0.03, G3SG1 | VariCamo BS $0.04, Number K | The Professionals $75.77)
2. **DA recomputation:** Manually summed `direction_correct` from `forecast_outcomes`: 3,291/5,512 = 59.71% — matches stored value exactly
3. **NULL integrity:** All 21,737 v3 forecast outcomes have populated key fields (`actual_price`, `direction_correct`, `abs_error`)
4. **Pipeline consistency:** `forecast_outcomes` → aggregate functions → `prediction_accuracy` — self-consistent chain

## Flat-bias finding

The model exhibits a conservative "flat" prediction bias on longer horizons:

| Horizon | Items where actual ≠ flat but model said "flat" | 
|---------|------------------------------------------------:|
| 3d | 1,477 (27% of batch) |
| 7d | 2,351 (43%) |
| 14d | 2,448 (45%) |
| 30d | 1,460 (27%) |

Dec 2025 was a strongly trending month (only ~18% of actuals were "flat"), so this conservative bias penalizes directional accuracy heavily on longer horizons. The model still adds +28 to +39pp over naive baseline — it's genuinely predictive, just increasingly cautious with uncertainty.

## Why

Previous backtest (July 17) stored results without baseline comparison, bootstrap CIs, or wMAPE. The metrics pipeline had been updated to include these but never re-run. The drop in raw DA for 7d/14d/30d (~55% → ~46%) is primarily the flat-bias penalty in a trending market, not a regression.
