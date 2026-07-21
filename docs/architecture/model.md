# Model Architecture

## Overview

Two-layer `ItemForecaster` containing **~144 LightGBM models** — a global ensemble and per-regime ensembles:

### Global models (always available)

```
3d horizon:  3 quantiles (p10, p50, p90) × 6 ensemble members
7d horizon:  3 quantiles (p10, p50, p90) × 6 ensemble members
14d horizon: 3 quantiles (p10, p50, p90) × 6 ensemble members
30d horizon: 3 quantiles (p10, p50, p90) × 6 ensemble members
```

72 total. Predictions averaged across ensemble members per quantile. p10/p90 provide the interval, p50 is the point prediction.

### Regime-specific models (trained per regime)

Separate ensembles for each detectable market regime (bear / range / bull), each with the same 72-model structure. Trained alongside global models when data volume permits. At prediction time, the current regime is detected and regime-specific models are preferred.

| Regime | Detection | Typical status |
|--------|-----------|----------------|
| bear | `market_return_30d < -3%` | Trained for 7d/14d, skipped for 3d/30d (insufficient val data) |
| range | `-3% ≤ return_30d ≤ 3%` | Fully trained |
| bull | `return_30d > 3%` | Fully trained |

**Model version:** `lgbm-v3-regime` (regime-enabled) / `lgbm-v3-global-only` (A/B comparison)
**Files:** `lgb_{horizon}d_q{q}_e{ei}.txt` (global) + `lgb_{horizon}d_q{q}_{regime}_e{ei}.txt` (regime) + `residual_{horizon}d_q{int(q*100)}.pkl` (Ridge residual models) + `meta.json` in `backend/models/saved_models/`

---

## Regime-Switching

### Motivation

Market regimes (bear/range/bull) have fundamentally different price dynamics. A single global model must average across all regimes, diluting signal in each. Separate per-regime models capture regime-specific patterns (e.g., mean-reversion in ranges, momentum in bulls).

### Training

In `train()`, after global models complete:

1. **Label assignment:** `_assign_regime_label()` tags each row by `market_return_30d` → bear/range/bull
2. **Per-regime training:** `_assign_regime_labels()` groups data by regime, calls `_train_horizon_ensemble()` with regime-specific subset
3. **Skip threshold:** regime skipped if <500 train or <50 val rows (bear often skipped for short horizons)
4. **Hyperparameters:** reuse global `tuned_params` (no separate Optuna per regime)
5. **Storage:** regime models saved to `self.regime_models[(regime, horizon, q)]` — parallel to `self.models[(horizon, q)]`

### Prediction

`predict()` detects current regime once via `_detect_current_regime()`:

```python
regime = _detect_current_regime(feature_row)
#     ^— "bear", "range", or "bull"
```

For each `(horizon, q)`:
- If `regime_models.get((regime, horizon, q))` exists → use it
- Otherwise → fall back to `models[(horizon, q)]`

Logs per-horizon usage ratio after prediction.

### Persistence

- Regime model filenames: `lgb_{horizon}d_q{q}_{regime}_e{ei}.txt`
- `meta.json` fields:
  - `trained_regimes`: list of regimes that have models saved
  - `regime_feature_cols`: feature columns per regime ensemble

### Training time impact

On the initial run (Mac, 400K rows, SKIP_CV):
- Global-only: ~30 min (53 min - 23 min regime overhead)
- Regime (range + bull, bear partial): +23 min
- Total: **53 min** for all 4 horizons

Speed concern is mitigated by the Monday retrain schedule + weekday predict-only pattern.

---

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

### Rarity Features (Added Jul 2026, refined Jul 2026)
Rarity ordinal and one-hot dummies (11 categories: base, consumer, industrial, milspec, restricted, classified, covert, high_grade, remarkable, exotic, extraordinary). Source: `price-archive/item-metadata.parquet` (8,691 items, 109 KB) with DB fallback. Permutation test confirmed strong causal signal (+10-12pp across all horizons).

**Removed after permutation test:** weapon_type one-hot dummies (22 cols) and weapon-type cross-sectional features (6 cols) — showed zero causal signal (shuffling changed accuracy by ≤0.05pp). Player count features (10 cols) also removed; earlier permutation test showed zero causal impact.

### Social Sentiment Features (Added 2026-07-19)
5 features sourced from the Reddit social sentiment collector (`collectors/social_sentiment.py`): `social_mentions_1d` (mention count past 24h), `social_mentions_7d` (rolling 7d), `social_mention_velocity` (1d→7d ratio), `social_sentiment_7d` (VADER compound score), `social_score_7d` (weighted sentiment × volume). Collector monitors 3 CS2 trading subreddits via old.reddit.com HTML (Reddit JSON API killed May 2026).

**Known limitation:** VADER is a 2014 general-purpose lexicon that scores CS2 market jargon ("BFK CW MW low float") as neutral — social features do not rank in top 20 by gain importance across any horizon. See `docs/changelog/2026-07-22-social-feature-audit.md`. Recommendation: replace VADER with ModernFinBERT (ONNX INT8).

### Feature Pruning
Correlation-based pruning at 0.95 threshold. Applied during training, pruned feature list saved to `meta.json`.

---

## Training Pipeline

### Hyperparameter Search
Optuna Bayesian (TPE sampler, MedianPruner). Replaced brute-force grid search Jul 2026.

| Before | After |
|--------|-------|
| Grid search: 81 combos (3⁴) | Optuna TPE: 15-50 Bayesian trials per quantile |
| Searched 4 params | Searches 6 (+ max_depth, min_data_in_leaf) |
| Fixed discrete values | Continuous ranges with log-uniform sampling |
| All trials to completion | MedianPruner kills bad trials early |

**Per-horizon trial budget (2026-07-20):** 15 trials proved insufficient for the 3d horizon's 6D search space (~40k combinations). The 3d model's chosen depth=3 was an artifact of under-exploration, not a signal ceiling. Added `N_TRIALS_MAP = {3: 50, 7: 15, 14: 15, 30: 15}`. With 50 trials, Optuna chose depth=5-8 across quantiles (3d p50: depth=5, L2=1.5).

**Optimized (2026-07-20):** 3d depth experiment showed 50 trials unnecessary — 20 trials + warm-start (depth=5, leaves=47, λ₂=1.5, lr=0.01) + horizon-aware bounds (max_depth 4-8, lambda_l2 0.5-2.0) reached same quality at 3 min instead of 7.5 min. Final: `N_TRIALS_MAP = {3: 20, 7: 15, 14: 15, 30: 15}`.

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
1460 days of backfilled data from Parquet archive. Changed from 365d→730d (Jul 2026) then 730d→1460d (2026-07-16) to improve long-horizon signal. Row count bounded by a pre-feature-engineering item-stratified subsample (max 400K rows) plus a post-feature-engineering safety cap (max 700K rows).

### Retrain Schedule
- Full retrain Monday: Optuna + 72 global ensemble models + ~180 regime ensemble models (~53 min total on Mac with SKIP_CV)
- Predict-only Tue-Sun: load saved models (global + regime both loaded)
- Drift-triggered: auto-retrain if directional accuracy drops below 60% on 7-day sliding window

### Boosting Strategy

Short horizons (3d, 7d) use standard **GBDT**. Longer horizons (14d, 30d) use **DART** (Dropout Additive Regression Trees) to reduce overfitting. DART uses 500 boost rounds (vs 1000 for GBDT) and adds three Optuna-tuned hyperparameters (`drop_rate`, `max_drop`, `skip_drop`).

### Residual Stacking

A **Ridge regression** model (`alpha=5.0`) trained on ensemble residuals to correct systematic bias for weak horizons (14d, 30d). Models saved as `residual_{horizon}d_q{int(q*100)}.pkl`.

### Forecast Blending

`FORECAST_BLEND_WEIGHT = 0.15` smooths predictions day-over-day, reducing direction flip-flopping by blending with the previous day's forecast.

### Parameters

| Parameter | Value |
|-----------|-------|
| `num_leaves` | 31 |
| `max_depth` | 5 |
| `min_data_in_leaf` | 15 |
| `min_gain_to_split` | 0.1 |
| `learning_rate` | 0.03 |
| `num_boost_round` | 1000 (GBDT) / 500 (DART) |
| `early_stopping` | 50 |
| `feature_fraction` | 0.7 |
| `bagging_fraction` | 0.7 |
| `lambda_l1` | 0.5 |
| `lambda_l2` | 0.5 |
| `boosting_type` | `gbdt` (3d/7d) / `dart` (14d/30d) |
| `drop_rate` | 0.05–0.3 (DART, Optuna-tuned) |
| `max_drop` | 10–100 (DART, Optuna-tuned) |
| `skip_drop` | 0.2–0.8 (DART, Optuna-tuned) |
| `max_bin` | 63 (was 255, halved for speed) |
| `feature_pre_filter` | `false` (LGB 4.6 compat) |
| `N_ENSEMBLES` | 6 (was 9, reduced for speed) |

### Training Time (Mac)
~53 min full retrain with regime models (~30 min global + ~23 min regime). ~17–20 min of Optuna on cold start. With SKIP_CV+HP reuse: ~30 min global + ~23 min regime.

---

## Prediction

### Regime detection
Current market regime detected from global `market_return_30d` mean before per-item prediction. If regime models exist for the detected regime, they are preferred over global models. Usage ratio logged per run.

### Eligibility
≥14 days of price history in Parquet archive. Filtered to items with STEAMCOMMUNITY backfill data (~5,542 items).

### Spike Smoothing
If latest price deviates >10% from 3d median, price is replaced with median value (~33% of items affected). Prevents outlier-driven predictions.

### Quantile Crossing Fix
When p10 > p50 or p90 < p50: impute average interval half-width from well-behaved items. Crossing rate logged when >1%.

### Sanitization
NaN/INF/negative forecast prices → clamped to `current_price` with `flat` direction and `low` confidence. High confidence downgraded for zero-volume items.

### Confidence
Binary: `high` or `low`. Threshold-calibrated per horizon from holdout predictions. Targets ≥80% directional accuracy within high-confidence bucket with minimum 5% coverage.

---

## Accuracy

### Summary

The model has two accuracy evaluation methods. The **production backtest** (below) is the canonical metric — it runs against 5,500 full-batch forecasts on real Parquet data with bootstrap CIs and baseline comparison. The walk-forward eval is a historical reference from an earlier 50-item test.

### Walk-Forward Evaluation (200 items, tuned params)

| Horizon | Directional Accuracy | vs 50% baseline | Interval Coverage | MAE |
|---------|:--------------------:|:---------------:|:-----------------:|:---:|
| **3d**  | 62.9%                | +12.9pp         | 86.1%             | $0.19 |
| **7d**  | 65.5%                | +15.5pp         | 86.8%             | $0.24 |
| **14d** | 67.7%                | +17.7pp         | 85.9%             | $0.32 |
| **30d** | 67.4%                | +17.4pp         | 83.1%             | $0.50 |

Measured via `walkforward_backtest.py` on 200 data-rich items, 26 expanding windows (60-day steps), ~109K samples per horizon. Uses tuned params from `meta.json`. Walkforward numbers are 5-13pp higher than production backtest because they test on data-rich items vs all 8,691 items (including sparse-history and dead items).

**Previous 50-item eval (archived):** 3d=61.4%, 7d=60.9%, 14d=60.2%, 30d=68.1% — Fixed LightGBM params (no ensemble).

### Historical Accuracy Timeline

| Stage | 7d Dir Acc | 30d Dir Acc | Notes |
|-------|:----------:|:-----------:|-------|
| Pre-audit (MA-crossover) | ~34% | ~34% | Random baseline (3-class) |
| After P1/P2 fixes | 70.9% | 72.5% | Leakage fix, returns target, NaN fix |
| After Jul '26 round 1 | 75.3% | 77.0% | Event decay, confidence calibration |
| After Jul '26 round 2 | 87.0% | 83.0% | **Buggy** — target inversion inflated |
| After target inversion fix | 61.1% | 65.8% | Genuine — 9-16pp above 50% baseline |
| After data quality + shift guard | 57.4% | 55.2% | Dead item filter, winsorization, 2026 exclusion |
| After 3d depth + walkforward | **52.8%** | **54.2%** | Production backtest (all 8,691 items) |

### Production Backtest Pipeline

The `backtest_accuracy.py` script evaluates mature forecasts from `item_forecasts` against actual prices from the Parquet archive (using the same multi-source voting as training). It stores aggregate metrics to `prediction_accuracy` and per-forecast outcomes to `forecast_outcomes`.

**Latest verified results (v3, 2026-07-20, 5,300–5,500 forecasts per horizon):**

| Horizon | DA | 95% CI | Baseline DA | vs Baseline | MAE | MAPE | wMAPE | IC |
|---------|---:|--------|------------:|------------:|----:|-----:|------:|--:|
| **3d** | **61.5%** | [60.2, 62.8] | 20.23% | **+41.27pp** | $0.73 | 33.2% | 30.5% | 47.1% |
| **7d** | 52.8% | [51.5, 54.1] | 17.42% | +35.38pp | $0.73 | 43.8% | 30.8% | 41.2% |
| **14d** | 55.7% | [54.3, 57.1] | 17.67% | +38.03pp | $0.74 | 51.9% | 30.9% | 49.8% |
| **30d** | 54.2% | [52.8, 55.6] | 18.00% | +36.20pp | $0.77 | 35.8% | 31.7% | 57.0% |

Baseline = persistence forecast (predicts zero change). Bootstrap CI = 95% percentile, 1,000 resamples. 3d high-confidence forecasts achieve ~78% accuracy vs ~43% low-confidence. Improvements from 3d depth experiment (+1.8pp on 3d) and walkforward-guided fixes.

**Metrics computed per horizon:**
- Point error: MAE, RMSE, MAPE, wMAPE (dollar-weighted), MAPE by price tier
- Directional: DA with bootstrap CI, interval coverage
- Baseline: persistence forecast DA/MAE, improvement over baseline (pp), skill_vs_baseline (Theil's U analog)
- Calibration: conf_gap_pp, conf_high_interval_cov, conf_calibration_error

### Known Limitations
- Walk-forward eval runs on 50 items only (not all 8,691)
- Fold variance is high (30d std=12.0%, range 42.5%–91.1%)
- Recent folds degrade during high market volatility
- Interval coverage drops in volatile periods
- Rarity features have strong causal signal (+10-12pp permutation test). Weapon-type one-hot and cross-sectional were removed — they showed zero causal signal despite the +0.66pp A/B bundle delta
- `_apply_multi_source_voting()` uses `groupby().apply()` on 5.5M rows — takes ~3-5 min. Could be vectorized for ~5s but voting logic is correct and this only affects training time, not accuracy
- **Flat-bias on longer horizons:** The model predicts "flat" too often on 7d/14d/30d when uncertainty is high. In a trending market (Dec 2025: only ~18% of actuals were flat), this penalizes raw DA heavily — 43% of 7d forecasts predicted flat when the item actually moved. The model still adds +29pp over baseline, but confidence calibration tuning could reduce this bias.
- **VADER social features not in top 20:** After 3 days of production data, none of the 5 social sentiment features rank in the top 20 by gain importance across 122 features. Root cause: VADER is a general-purpose lexicon that scores CS2 market jargon ("BFK CW MW low float") as neutral. Recommendation: replace with ModernFinBERT (ONNX INT8, ~88% accuracy on financial text).

### Data Quality Audit (2026-07-17)
A comprehensive audit of the 9.8M-row training set revealed three data quality issues:

| Issue | Scope | Impact |
|-------|-------|--------|
| **Dead items at Steam floor** ($0.03-0.04) | 2,936 items, 4.1M rows (41.5%) | Zero signal — dilutes model, wastes gradient steps |
| **Corrupt price jumps** (>1000%, revert next day) | 11,044 jumps across 905 items | Pollutes gradient estimates with false targets |
| **Incomplete 2026 backfill** (ends Mar 29) | ~400K rows | Distribution shift: train on Jan-Mar, predict on Jul+ |

All three issues were fixed in the 2026-07-17 changelog. Full details in `docs/changelog/2026-07-17-data-quality-audit-and-fixes.md`.

---

## Horizon Selection

| Horizon | Status | Rationale |
|---------|--------|-----------|
| 1d | ❌ Rejected | Too noisy — day-to-day CS2 price action dominated by random walk |
| 3d | ✅ Active | Short-term momentum, smooths weekend gaps |
| 7d | ✅ Active | Primary horizon, strongest mid-term signal |
| 14d | ✅ Active | Natural midpoint, many CS2 cycles run ~2 weeks |
| 30d | ✅ Active | Longest horizon, benefits most from 1460d training window |
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
| Supply-side features (rarity one-hot) | 2026-07-15 | Low (+0.66pp bundle, refined to rarity-only Jul 16) |
| Weapon-type features removed (22 one-hot + 6 cross-sectional) | 2026-07-16 | Zero causal signal (permutation test) |
| Multi-source outlier voting | 2026-07-16 | Medium — protects live inference from corrupted `current_price` |
| Walk-forward backtest (`walkforward_backtest.py`) | 2026-07-20 | Medium — reusable 200-item, 26-fold walk-forward evaluation |
| HF CS2 dataset merge (69.2M hourly rows) | 2026-07-20 | Medium — filled Mar 30–Apr 15 gap in Parquet archive |
| Social sentiment features (5 cols, VADER) | 2026-07-19 | Low — not in top 20 gain importance; VADER ineffective on CS2 jargon |
| 3d depth experiment (`N_TRIALS_MAP[3]: 50→20`) | 2026-07-20 | Medium — +1.8pp on 3d, 3 min vs 7.5 min Optuna |
| 3-seed ensemble per quantile | Jul 2026 | Medium |
| Binary confidence (dropped medium bucket) | Jul 2026 | Low |
| CatBoost ensemble tested and removed | 2026-07-13 | Low (degraded accuracy) |
| Recency mismatch fix (365d → 365d aligned) | Jul 2026 | Medium |
| 60d rolling features + 730d training window | 2026-07-12 | Medium |
| Concept drift monitoring | Jul 2026 | Medium |
| **Regime-switching models** (bear/range/bull per horizon) | **2026-07-18** | **High** — captures regime-specific price dynamics |
| **DART boosting for 14d/30d** | Done | **Medium** — reduces overfitting on noisy long horizons |
| **Residual stacking** (Ridge on LGB residuals) | Done | **Medium** — corrects systematic bias for weak horizons |
| **Forecast blending** (`FORECAST_BLEND_WEIGHT=0.15`) | Done | **Low** — reduces daily direction flip-flopping |
| Prediction sanity checks (clamp, zero-volume) | 2026-07-12 | Low |
| 41 unit tests for forecaster | 2026-07-12 | Foundation |
| **Dead item filter** (remove $0.03 floor items) | 2026-07-17 | High — +2-5pp estimated, 41% less training noise |
| **Target winsorization** (±500% clip) | 2026-07-17 | High — neuters 11K corrupt price jumps |
| **Corrupt item flagging** (exclude 151 worst) | 2026-07-17 | Medium — removes API corruption from training |
| **Sample weighting** (by price variance) | 2026-07-17 | Medium — down-weights flat items, up-weights movers |
| **2026 shift guard** (exclude incomplete year) | 2026-07-17 | Low — closes train/predict distribution gap |

### What NOT To Do
- **Do not replace with a fine-tuned LLM** — worse at numerical time series, slower, harder to retrain
- **Do not add neural forecasting models** (N-BEATS, PatchTST) unless accuracy plateaus and GPU training is manageable
- **Do not revisit CatBoost** — tested Jul 2026, degraded accuracy by 18-20pp
- **Do not ship VADER-based social features as-is** — 2014 general-purpose lexicon scores CS2 jargon as neutral. Replace with ModernFinBERT (ONNX INT8) first

---

## Files Reference

| File | Lines | Role |
|------|-------|------|
| `backend/models/forecaster.py` | ~3,500 | Core ML: ItemForecaster, feature engineering, training, predict, regime-switching |
| `backend/models/steam_types.py` | 130 | Steam type field parser (rarity + weapon_type extraction) |
| `backend/scripts/forecast_prices.py` | 283 | Entry point: train + predict pipeline, `--compare-regime` A/B mode |
| `backend/scripts/backtest_accuracy.py` | 399 | Mature forecast backtesting (production) |
| `backend/scripts/walkforward_backtest.py` | — | Standalone walk-forward evaluation (200 items, 26 folds) |
| `backend/scripts/evaluate_forecaster.py` | 366 | Legacy walk-forward evaluation (archived) |
| `backend/scripts/optuna_3d_search.py` | — | Optimized 3d horizon Optuna search (200 items, ~58s) |
| `backend/scripts/3d_depth_experiment.py` | — | 3d depth hyperparameter depth experiment |
| `backend/scripts/merge_hf_dataset.py` | — | Merge HuggingFace CS2 dataset into Parquet archive |
| `backend/scripts/backfill_supply_metadata.py` | — | Backfill supply metadata from catalog → Parquet + DB |
| `backend/collectors/social_sentiment.py` | — | Reddit sentiment collector (3 subreddits, VADER, 6-hourly) |
| `backend/tests/test_forecaster.py` | 1,118 | 69+ unit tests (global + regime-switching) |
| `price-archive/item-metadata.parquet` | 8,691 rows | Supply metadata cache (rarity, weapon_type per item) |
