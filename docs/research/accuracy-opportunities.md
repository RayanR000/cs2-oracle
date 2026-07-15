# Prediction Accuracy Improvement Opportunities

Date: 2026-07-14

## Current Architecture

- **Model**: LightGBM quantile regression — 4 horizons (3d, 7d, 14d, 30d) × 3 quantiles (p10, p50, p90) × 3 ensemble seeds = 36 models
- **HP Optimization**: Optuna Bayesian (15 trials per quantile), expanding-window CV
- **Feature count**: ~45-120 after correlation pruning (threshold 0.95)
- **Feature categories**: Price technicals (lags, rolling stats, Bollinger, RSI, MACD, support/resistance, volume), temporal (cyclic time features), events (5 types with exponential decay), cross-sectional (market returns, regime flags)
- **Drift threshold**: 60% directional accuracy on 7-day sliding window
- **Confidence calibration**: 80% target accuracy, min 5% coverage
- **Training data**: 730 days backfilled from Parquet archive (2013-2026), max 200K rows
- **Retrain schedule**: Full retrain Mondays, predict other days, auto-retrain on drift

---

## 1. Feature Engineering

| Feature | Rationale | Est. Impact | Effort |
|---------|-----------|-------------|--------|
| Category/collection features (same weapon group, collection, case) | Items in same category move together — category returns, volatility | 2-5pp | Low |
| Steam active listing count (vs. trade volume) | Supply indicator; listings/trades ratio = liquidity | 3-6pp | Medium |
| Item liquidity score (volume churn ratio) | Low-liquidity items have larger price impact per trade | 2-4pp | Low |
| Steam player count | Core demand driver — correlates with market activity | 2-4pp | Low |
| Tournament/major timeline + results | Skins of winning teams/players spike in price | 3-8pp | Medium |
| Float/wear distribution features | Different wears behave as separate markets | 1-3pp | Medium |
| Price clustering / round-number resistance | Psychological price levels ($10, $50, $100) | 1-2pp | Low |
| Post-spike mean reversion speed | How quickly items revert after volume spikes | 2-3pp | Low |
| Listing density (spread between min ask and max bid if available) | Market depth signal | 2-4pp | High |

---

## 2. Model Architecture

| Approach | Expected Benefit | Complexity |
|----------|-----------------|------------|
| **Linear/Ridge head ensemble** — hybrid tree + linear model captures both non-linear and trend-following regimes | 2-5pp accuracy | Low |
| **Multi-horizon joint training** — predict all 4 horizons in one model; short-term signal informs long-term | 1-3pp, fewer crossing issues | Medium |
| **Regime-switching models** — separate LGBM for bull/bear/range regimes (detected via market_regime feature as classifier) | 3-8pp during volatile periods | Medium |
| **Expand ensemble to 7-10 seeds** with column subsampling variation (not just different random seeds) | 1-2pp, more robust | Low |
| **N-BEATS or Temporal Fusion Transformer** as secondary stack — tree + neural net ensemble | 3-8pp | High |
| **Hierarchical forecast** — predict market → category → item, constrain item forecasts to sum to category | 2-4pp | High |

---

## 3. Training Pipeline

| Change | Impact | Effort |
|--------|--------|--------|
| Increase Optuna trials from 15 to 50-100 | 2-5pp | Low (compute only) |
| Per-cluster models — cluster items by volatility/volume/liquidity, train specialized models | 3-8pp for tail/low-volume items | Medium |
| Time-decayed loss weighting (weight = α^(days_ago), α=0.99) | 2-4pp, adapts to recent regime | Low |
| Adversarial validation between train and serving data | Better drift detection | Medium |
| Rolling retrain on any day accuracy degrades (not just triggered at 60%) | 2-3pp sustained | Low |
| Learning rate warmup + schedule decay | 1-2pp | Low |
| Gradient-based feature selection (SHAP importance pruning) | Simplifies model, prevents overfit | Low |

---

## 4. Post-Processing & Calibration

| Change | Impact | Effort |
|--------|--------|--------|
| Directional smoothing — EMA on predicted direction to reduce daily flip-flopping | 1-2pp, more stable | Low |
| 4-tier confidence instead of binary (high/medium/low/very-low) | Better risk stratification | Low |
| Ensemble variance as confidence signal | More calibrated uncertainty | Low |
| Conformal prediction on p10/p90 intervals | Better coverage guarantees | Medium |
| Forecast blending — blend current prediction with previous day's at small weight | Reduces jumpiness, 1-2pp | Low |

---

## 5. Data Quality

| Change | Impact | Effort |
|--------|--------|--------|
| Multi-source outlier voting — if 5/7 sources agree, downweight outliers | 2-4pp | Low |
| Intraday high/low price range per source per day | Volatility signal, 1-3pp | Medium |
| Gap-fill with interpolation instead of forward-fill | More continuous signal | Low |
| Source reliability scoring — weight each source by historical accuracy | 1-3pp | Medium |
| Consistent timestamps across sources (align to UTC hour) | Prevents stale-data comparisons | Low |

---

## 6. External Data Sources

| Source | Signal | Difficulty |
|--------|--------|------------|
| [SteamCharts](https://steamcharts.com/) API | Player count trends | Low |
| Twitch/YouTube CS2 category metrics | Hype cycles, content trends | Medium |
| Liquipedia tournament schedule + results | Major/event anticipation & reaction | Medium |
| Reddit r/GlobalOffensive, r/csgomarketforum | Sentiment (early hype) | High |
| Steam Community Market listing count API | Supply depth | Medium |

---

## Priority Order

1. **⚠️ Completed: Supply-side features** — rarity + weapon_type + weapon-type cross-sectional. Actual impact +0.66pp avg (below 3-6pp estimate).
2. **⚠️ Completed: Player count** — zero causal impact (permutation test).
3. **🔥 Top remaining: Event decay optimization** + **Multi-horizon joint training** — moderate effort, 2-4pp potential each.
4. **Listing volume / supply depth** — new external data collection needed.
5. **Regime-switching models / per-cluster models** — moderate-to-high effort, 3-8pp potential.
6. **Directional smoothing + ensemble expansion** — quick post-processing wins.
7. **Neural model hybrid (N-BEATS/TFT)** — longer-term investment.

---

## Notes

- CatBoost was tested and removed (Jul 2026) — degraded accuracy by 18-20pp — do not revisit
- Trend analyzer was deprecated and removed (Jul 2026)
- Grid search replaced by Optuna Bayesian (Jul 2026)
- Model version is `lgbm-v2`; any architecture change should increment to `lgbm-v3`
- All changes must pass `test_forecaster.py` (28+ tests)
- Production models stored in `backend/models/saved_models/` — can serve multiple model versions simultaneously
