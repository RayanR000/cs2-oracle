# Model Architecture

## Overview

Single `ItemForecaster` containing **36 LightGBM models** for quantile regression:

```
3d horizon:  3 quantiles (p10, p50, p90) × 3 seeds (42, 73, 91)
7d horizon:  3 quantiles (p10, p50, p90) × 3 seeds (42, 73, 91)
14d horizon: 3 quantiles (p10, p50, p90) × 3 seeds (42, 73, 91)
30d horizon: 3 quantiles (p10, p50, p90) × 3 seeds (42, 73, 91)
```

Predictions are averaged across seeds per quantile. p10/p90 provide the interval, p50 is the point prediction.

**Model version:** `lgbm-v2`
**Last trained:** 2026-07-13 (full retrain)
**Files:** 36 `lgb_*.txt` + `meta.json` in `backend/models/saved_models/`

---

## Feature Engineering (110+ features)

### Price Features
Lags (1/3/7/14/30/60d), returns (winsorized ±500%), rolling stats (7/14/20/30/60d windows), log returns, autocorrelation proxies, price acceleration.

### Technical Indicators
Bollinger Bands (20d: bb_upper, bb_lower, bb_pct_b, bb_width), RSI (14d), MACD (line, signal, histogram), support/resistance distances, high/low range.

### Volume Features
Lags, log-change (avoids division-by-zero), Z-score relative to 30d mean, volume-price confirmation signals.

### Temporal Features
Day-of-week/month/quarter/year, sin/cos cyclic encoding, weekend flag, item age. Added in Jul 2026 to replace raw integer encoding which LightGBM treated as ordinal (Dec→Jan looked like gap of 11).

### Event Features
5 event types (major, operation, case_drop, update, game_update) with exponential decay weighting (`exp(-days_since / decay_constant)`), event density counts (30d/90d), events-next-30d count.

### Cross-Sectional Features
Market return per lag, item return vs market, market volatility/volume/regime flags (bull/bear/range), item volume vs market.

### Supply-Side Features (Added Jul 2026)
Rarity ordinal and one-hot dummies (11 categories: base, consumer, industrial, milspec, restricted, classified, covert, high_grade, remarkable, exotic, extraordinary), weapon type one-hot dummies (22 categories: rifle, pistol, smg, shotgun, sniper, machinegun, knife, glove, case, sticker, graffiti, musickit, charm, agent, patch, collectible, equipment, key, pass, tool, tag, gift). Source: `price-archive/item-metadata.parquet` (8,691 items, 109 KB) with DB fallback.

### Weapon-Type Cross-Sectional Features (Added Jul 2026)
Per-date weapon-type group returns, item return vs weapon-type mean, weapon-type volatility and volume signals. Parallels the market-level cross-sectional features but computed per weapon_type group. Uses `wt_` prefix to avoid collision with identity features (e.g., `wt_group_return_7d`).

### Feature Pruning
Correlation-based pruning at 0.95 threshold. Applied during training, pruned feature list saved to `meta.json`.

---

## Training Pipeline

### Hyperparameter Search
Optuna Bayesian (TPE sampler, MedianPruner). Replaced brute-force grid search Jul 2026.

| Before | After |
|--------|-------|
| Grid search: 81 combos (3⁴) | Optuna TPE: 15-30 Bayesian trials per quantile |
| Searched 4 params | Searches 6 (+ max_depth, min_data_in_leaf) |
| Fixed discrete values | Continuous ranges with log-uniform sampling |
| All trials to completion | MedianPruner kills bad trials early |

**Bug fix (Jul 2026):** `_optuna_search_params` originally hardcoded `alpha=0.5`. The `quantile` parameter was never received — p10 and p90 models were tuned at alpha=0.5. Fixed by adding `quantile` to method signature.

### Validation
Expanding-window cross-validation (5 folds, 120-day steps):

```
Fold 1: train [0..200d] → val [201..221d]
Fold 2: train [0..320d] → val [321..341d]
Fold 3: train [0..440d] → val [441..461d]
Fold 4: train [0..560d] → val [561..581d]
Fold 5: train [0..680d] → val [681..701d]
```

Confidence calibration uses pooled out-of-fold predictions across all folds. Added Jul 2026 (was single 21-day holdout, which was sensitive to specific events in that window).

### Training Window
730 days of backfilled data from Parquet archive. Changed from 365d to 730d Jul 2026 to improve 30d horizon signal. Max 200K rows subsampled for speed.

### Retrain Schedule
- Full retrain: Monday (Optuna + 36 ensemble models)
- Predict-only: Tue-Sun (load saved models)
- Drift-triggered: auto-retrain if directional accuracy drops below 60% on 7-day sliding window

### Parameters

| Parameter | Value |
|-----------|-------|
| `num_leaves` | 31 |
| `max_depth` | 5 |
| `min_data_in_leaf` | 15 |
| `min_gain_to_split` | 0.1 |
| `learning_rate` | 0.03 |
| `num_boost_round` | 1000 |
| `early_stopping` | 50 |
| `feature_fraction` | 0.7 |
| `bagging_fraction` | 0.7 |
| `lambda_l1` | 0.5 |
| `lambda_l2` | 0.5 |

### Training Time (M4)
~30 min full retrain (~17 min Optuna + CV + ensemble).

---

## Prediction

### Eligibility
≥14 days of price history in Parquet archive. Filtered to items with STEAMCOMMUNITY backfill data (~5,542 items).

### Spike Smoothing
If latest price deviates >10% from 3d median, price is replaced with median value (~33% of items affected). Prevents outlier-driven predictions.

### Quantile Crossing Fix
When p10 > p50 or p90 < p50: impute average interval half-width from well-behaved items. Crossing rate logged when >1%.

### Sanitization
NaN/INF/negative forecast prices → clamped to `current_price` with `flat` direction and `low` confidence. High confidence downgraded for zero-volume items.

### Confidence
Binary: `high` or `low`. Threshold-calibrated per horizon from CV out-of-fold predictions. Targets ≥80% directional accuracy within high-confidence bucket with minimum 5% coverage.

---

## Accuracy

### Current (Post-Fix, Genuine)

| Horizon | Directional Accuracy | vs 50% baseline | Interval Coverage | MAE |
|---------|:--------------------:|:---------------:|:-----------------:|:---:|
| **3d**  | **61.4%**            | **+11.4pp**     | 85.7%             | $0.20 |
| **7d**  | 60.9%                | +10.9pp         | 86.2%             | $0.25 |
| **14d** | **60.2%**            | **+10.2pp**     | 86.1%             | $0.34 |
| **30d** | 68.1%                | +18.1pp         | 82.6%             | $0.53 |

Measured via walk-forward evaluation on 50 items, 26 expanding windows (60-day steps), ~27k samples per horizon. Fixed LightGBM params (no ensemble — conservative estimate). Supply-side features enabled (rarity one-hot, weapon_type one-hot, weapon-type cross-sectional). Lift: +0.66pp avg vs control (3d: +1.92pp, 7d: -0.16pp, 14d: +0.79pp, 30d: +0.08pp).

### Historical Accuracy Timeline

| Stage | 7d Dir Acc | 30d Dir Acc | Notes |
|-------|:----------:|:-----------:|-------|
| Pre-audit (MA-crossover) | ~34% | ~34% | Random baseline (3-class) |
| After P1/P2 fixes | 70.9% | 72.5% | Leakage fix, returns target, NaN fix |
| After Jul '26 round 1 | 75.3% | 77.0% | Event decay, confidence calibration |
| After Jul '26 round 2 | 87.0% | 83.0% | **Buggy** — target inversion inflated |
| After target inversion fix | **61.1%** | **65.8%** | **Genuine** — 9-16pp above 50% baseline |

### Known Limitations
- Walk-forward eval runs on 50 items only (not all 8,691)
- Fold variance is high (30d std=12.0%, range 42.5%–91.1%)
- Recent folds degrade during high market volatility
- Interval coverage drops in volatile periods
- Supply-side features have modest impact (+0.66pp avg) — existing cross-sectional features already capture much of the signal

---

## Horizon Selection

| Horizon | Status | Rationale |
|---------|--------|-----------|
| 1d | ❌ Rejected | Too noisy — day-to-day CS2 price action dominated by random walk |
| 3d | ✅ Active | Short-term momentum, smooths weekend gaps |
| 7d | ✅ Active | Primary horizon, strongest mid-term signal |
| 14d | ✅ Active | Natural midpoint, many CS2 cycles run ~2 weeks |
| 30d | ✅ Active | Longest horizon, benefits most from 730d training window |
| 60d | ❌ Not yet | Would require more data and richer long-term features |

---

## Historical Audit Reference

### Critical Bug: Target Inversion (2026-07-12)

`prepare_targets` was looking up prices `horizon` days **ago** instead of **ahead**. The model learned that `return_7d` (a feature) ≈ `target_return_7d` (the target), producing deceptively high 86-88% accuracy. Fixed by reversing the merge direction: each row looks up price at `date + horizon` using a backward date shift.

**Impact:** Real accuracy is ~60-66% (9-16pp above 50% baseline) instead of the illusory 86-88%.

### Key Fixes Applied

| Fix | Date | Impact |
|-----|------|--------|
| Remove `daily_analysis` feature leakage | Jul 2026 | High |
| Change target from price level to returns | Jul 2026 | High |
| Target inversion fix | 2026-07-12 | Critical |
| NaN imputation (per-feature medians) | Jul 2026 | High |
| Add RSI, MACD, Bollinger %B features | Jul 2026 | High |
| Walk-forward validation split | Jul 2026 | High |
| Optuna Bayesian HP search | Jul 2026 | Medium |
| Cyclical temporal encoding | 2026-07-12 | Medium |
| Feature medians persisted to meta.json | 2026-07-12 | Medium |
| Expanding-window CV | 2026-07-13 | Medium |
| Cross-sectional/market-regime features | Jul 2026 | Medium |
| Event decay weighting (not hardcoded 999) | Jul 2026 | Medium |
| Feature pruning (correlation 0.95) | Jul 2026 | Medium |
| Supply-side features (rarity, weapon_type, wt cross-sectional) | 2026-07-15 | Low (+0.66pp) |
| 3-seed ensemble per quantile | Jul 2026 | Medium |
| Binary confidence (dropped medium bucket) | Jul 2026 | Low |
| CatBoost ensemble tested and removed | 2026-07-13 | Low (degraded accuracy) |
| Recency mismatch fix (365d → 365d aligned) | Jul 2026 | Medium |
| 60d rolling features + 730d training window | 2026-07-12 | Medium |
| Concept drift monitoring | Jul 2026 | Medium |
| Prediction sanity checks (clamp, zero-volume) | 2026-07-12 | Low |
| 41 unit tests for forecaster | 2026-07-12 | Foundation |

### What NOT To Do
- **Do not replace with a fine-tuned LLM** — worse at numerical time series, slower, harder to retrain
- **Do not add neural forecasting models** (N-BEATS, PatchTST) unless accuracy plateaus and GPU training is manageable
- **Do not revisit CatBoost** — tested Jul 2026, degraded accuracy by 18-20pp

---

## Files Reference

| File | Lines | Role |
|------|-------|------|
| `backend/models/forecaster.py` | 2,012 | Core ML: ItemForecaster class, feature engineering, training, predict |
| `backend/models/steam_types.py` | 130 | Steam type field parser (rarity + weapon_type extraction) |
| `backend/scripts/forecast_prices.py` | 193 | Entry point: train + predict pipeline |
| `backend/scripts/evaluate_forecaster.py` | 366 | Walk-forward accuracy evaluation |
| `backend/scripts/backtest_accuracy.py` | 360 | Mature forecast backtesting |
| `backend/scripts/backfill_supply_metadata.py` | — | Backfill supply metadata from catalog → Parquet + DB |
| `backend/scripts/ab_test_supply_side.py` | — | A/B test: with vs without supply-side features |
| `backend/tests/test_forecaster.py` | — | 46 unit tests |
| `price-archive/item-metadata.parquet` | 8,691 rows | Supply metadata cache (rarity, weapon_type per item) |
