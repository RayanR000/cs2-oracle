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

| Feature | Rationale | Est. Impact | Calibrated | Effort |
|---------|-----------|-------------|-------------|--------|
| Category/collection features (same weapon group, collection, case) | Items in same category move together — category returns, volatility | 2-5pp | **0pp** ✅ tested | Low |
| Steam active listing count (vs. trade volume) | 🛑 **DROPPED** — supply-side, but only change/velocity variant is directionally predictive and needs 30d history/paid backfill; free source too slow. See §1 DECISION. | 3-6pp est | 0pp pursued | — |
| Item liquidity score (volume churn ratio) | Low-liquidity items have larger price impact per trade | 2-4pp | 1-2pp | Low |
| Steam player count | Core demand driver — correlates with market activity | 2-4pp | **0pp** ✅ tested | Low |
| Tournament/major timeline + results | Skins of winning teams/players spike in price | 3-8pp | 1-3pp | Medium |
| Float/wear distribution features | Different wears behave as separate markets | 1-3pp | 1-2pp | Medium |
| Price clustering / round-number resistance | Psychological price levels ($10, $50, $100) | 1-2pp | 0-1pp | Low |
| Post-spike mean reversion speed | How quickly items revert after volume spikes | 2-3pp | 0-1pp | Low |
| Listing density (spread between min ask and max bid if available) | Market depth signal | 2-4pp | 1-2pp | High |

> ⚠️ **CRITICAL DISTINCTION — "listing volume" ≠ "trade volume".**
> The supply-depth features above (active *sell_listings* count, listing density, supply-to-volume ratio) are **supply-side** signals and are the genuinely novel remaining input. They are **NOT** the same as **trade volume** (units *sold*), which was audited on 2026-07-16 and found to add **ZERO predictive lift** — every trade-volume feature correlates with forward returns at **|r| < 0.002** (statistical noise). See `docs/research/volume-data.md:25-29` and `docs/references/data-sources.md:75-83`. Trade volume's only value in this stack is confidence/liquidity weighting, never forecasting. If a future contributor reads "listing volume" and adds *sales* volume, that is the mistake to avoid — use `sell_listings` from the `supply_scraper` / `supply_snapshots` table, not traded-volume.
>
> 🛑 **DECISION (2026-07-16): Supply depth is DROPPED as a prediction-accuracy improvement.** Rationale: (1) only the *change/velocity* variant (`supply_change_7d`, `supply_listings_zscore`) is mechanistically predictive of direction — the *level* feature is a liquidity signal that does not move directional accuracy (CS2Cap: "liquidity is a tradability signal, not a price forecast"); (2) the change features require 30+ days of `supply_snapshots` history or a paid historical backfill (CS2Cap candles `q`); (3) the only free source is the Steam full-catalog scrape (~115 min/day) — deemed too slow/high-effort, and no free bulk listing-count source exists; paid APIs (CSMarketCap $9.99/mo, CS2Cap $19/mo) were rejected. Expected lift was only ~+1-2pp directional. Remaining accuracy work shifts to model architecture (regime-switching, Ridge head) on existing data. The `_add_supply_depth_features` code remains but is excluded from the accuracy roadmap.

---

## 2. Model Architecture

| Approach | Expected Benefit | Calibrated | Complexity |
|----------|-----------------|------------|------------|
| **Linear/Ridge head ensemble** — hybrid tree + linear model captures both non-linear and trend-following regimes | 2-5pp | 1-2pp | Low |
| **Multi-horizon joint training** — predict all 4 horizons in one model; short-term signal informs long-term | 1-3pp | 1-2pp | Medium |
| **Regime-switching models** — separate LGBM for bull/bear/range regimes (detected via market_regime feature as classifier) | 3-8pp | 2-4pp | Medium |
| **Expand ensemble to 7-10 seeds** with column subsampling variation (not just different random seeds) | 1-2pp | 1-2pp | Low |
| **N-BEATS or Temporal Fusion Transformer** as secondary stack — tree + neural net ensemble | 3-8pp | 1-3pp | High |
| **Hierarchical forecast** — predict market → category → item, constrain item forecasts to sum to category | 2-4pp | 1-2pp | High |

---

## 3. Training Pipeline

| Change | Impact | Calibrated | Effort |
|--------|--------|------------|--------|
| Increase Optuna trials from 15 to 50-100 | 2-5pp | 0.5-1pp | Low (compute only) |
| Per-cluster models — cluster items by volatility/volume/liquidity, train specialized models | 3-8pp | 1-3pp | Medium |
| Time-decayed loss weighting (weight = α^(days_ago), α=0.99) | 2-4pp | 1-2pp | Low |
| Adversarial validation between train and serving data | Better drift detection | Low | Medium |
| Rolling retrain on any day accuracy degrades (not just triggered at 60%) | 2-3pp | 1-2pp | Low |
| Learning rate warmup + schedule decay | 1-2pp | 0-1pp | Low |
| Gradient-based feature selection (SHAP importance pruning) | Simplifies model, prevents overfit | Low | Low |

---

## 4. Post-Processing & Calibration

| Change | Impact | Calibrated | Effort |
|--------|--------|------------|--------|
| Directional smoothing — EMA on predicted direction to reduce daily flip-flopping | 1-2pp | 1-2pp | Low |
| 4-tier confidence instead of binary (high/medium/low/very-low) | Better risk stratification | Low | Low |
| Ensemble variance as confidence signal | More calibrated uncertainty | Low | Low |
| Conformal prediction on p10/p90 intervals | Better coverage guarantees | Medium | Medium |
| Forecast blending — blend current prediction with previous day's at small weight | Reduces jumpiness, 1-2pp | 1-2pp | Low |

---

## 5. Data Quality

| Change | Impact | Calibrated | Effort |
|--------|--------|------------|--------|
| Multi-source outlier voting — if 5/7 sources agree, downweight outliers | 2-4pp | 2-4pp (keep) | Low |
| Intraday high/low price range per source per day | Volatility signal, 1-3pp | 1-2pp | Medium |
| Gap-fill with interpolation instead of forward-fill | More continuous signal | Low | Low |
| Source reliability scoring — weight each source by historical accuracy | 1-3pp | 1-2pp | Medium |
| Consistent timestamps across sources (align to UTC hour) | Prevents stale-data comparisons | Low | Low |

---

## 6. External Data Sources

| Source | Signal | Difficulty |
|--------|--------|------------|
| [SteamCharts](https://steamcharts.com/) API | Player count trends | Low |
| Twitch/YouTube CS2 category metrics | Hype cycles, content trends | Medium |
| Liquipedia tournament schedule + results | Major/event anticipation & reaction | Medium |
| Reddit r/GlobalOffensive, r/csgomarketforum | Sentiment (early hype) | High |
| Steam Community Market listing count API | 🛑 **DROPPED** — supply depth not pursued (2026-07-16); paid bulk APIs (CSMarketCap $9.99, CS2Cap $19) rejected | — |

---

## Priority Order

1. **✅ Completed: Supply-side features** — rarity one-hot kept (+10-12pp causal signal within model). Weapon_type (22 cols) and cross-sectional (6 cols) removed — zero causal signal. Bundle A/B test: +0.66pp avg.
2. **✅ Completed: Player count** — **Removed**. +3.0pp A/B but **0pp** causal (extra model capacity inflation). Permutation test: shuffled = real to within 0.03pp.
3. **✅ Completed: Event decay optimization** — **0pp**. Coordinate-wise tau grid search found defaults were already optimal. Walk-forward A/B: "optimal" taus degraded by -0.57pp.
4. **✅ Completed: Auto-prune (permutation-based feature validation)** — prevents overfit by removing feature groups that fail permutation test.
5. **✅ Completed: Multi-source outlier voting** — **0pp on training, essential for inference**. 99.6% of training data is single-source STEAMCOMMUNITY backfill; voting only affects 0.4% of rows. At inference, 96.4% of items see >0.5% price correction on current_price. See `docs/2026-07-14-remaining-accuracy-improvements.md` for full analysis.
### Dropped
- 🛑 **Supply depth (active `sell_listings` count)** — **DROPPED (2026-07-16).** Was the top remaining ROI. Dropped because: only the *change/velocity* variant is directionally predictive (the level is a liquidity signal, not a forecast); change features need 30+ days of `supply_snapshots` history or a paid backfill; the only free source is a ~115 min/day Steam scrape (too slow) and no free bulk listing-count source exists; paid APIs were rejected. Expected lift was only ~+1-2pp. Full rationale in the §1 DECISION note.

### Remaining (re-prioritized — regime-switching is now the top item)
6. **Regime-switching models** — moderate effort, 2-4pp during volatile periods (lower averaged). **Now the top remaining item.**
7. **Multi-horizon joint training** — moderate effort, 1-2pp potential.
8. **Quality spread / cross-wear features** — genuinely new signal, 1-2pp potential.
9. **Directional smoothing + ensemble expansion** — quick post-processing wins.

---

## Notes

- CatBoost was tested and removed (Jul 2026) — degraded accuracy by 18-20pp — do not revisit
- **Do NOT add trade/sales volume as a predictive feature** — audited 2026-07-16, |r| < 0.002 with forward returns (0pp). Supply depth (`sell_listings`) was also evaluated and **dropped (2026-07-16)** as an accuracy improvement (see §1 DECISION): only its change/velocity variant is mechanistically predictive but requires 30+ days of history or a paid backfill, which was not pursued.
- Trend analyzer was deprecated and removed (Jul 2026)
- Grid search replaced by Optuna Bayesian (Jul 2026)
- Model version is `lgbm-v2`; any architecture change should increment to `lgbm-v3`
- All changes must pass `test_forecaster.py` (28+ tests)
- Production models stored in `backend/models/saved_models/` — can serve multiple model versions simultaneously

---

## Reality Check — Calibrating Estimates

Every completed feature group was measured. The pattern is consistent:

| Feature | Estimate | Actual | Calibration Factor |
|---------|:-------:|:------:|:------------------:|
| Supply-side bundle (rarity + weapon_type) | +3-6pp | **+0.66pp** | ~15-20% of estimate |
| Player counts | +2-4pp | **0pp** (spurious +3pp A/B) | — |
| Event decay optimization | Small | **0pp** | — |
| CatBoost | not est. | **-18 to -20pp** | — |
| Multi-source outlier voting | +2-4pp | **0pp train / essential inference** | Pre-backfill estimate; 99.6% training data now single-source |

### Root Causes

1. **Extra capacity inflation.** Adding more features gives LightGBM more leaves to split on, inflating validation accuracy by 1-4pp even when the features have zero causal signal. Player counts showed +3pp A/B → 0pp permutation. **Always pair A/B tests with permutation tests.**

2. **Existing features capture most signal.** Price technicals (lags, returns, rolling stats, Bollinger, RSI, MACD) + cross-sectional (market returns, regime) → ~55-60pp directional accuracy. Rarity adds ~+10pp causal within the model, but the marginal gain of adding it to the baseline was only ~+0.5pp because the model partially compensates. **Past ~70 features, each new group delivers 30-50% of the initial estimate.**

3. **Estimates assume independent signal. They're not independent.** When features are correlated (and most market features are), the marginal gain of any new feature shrinks as the set grows.

### Calibrated Rule

For any new feature group added to the current ~70-feature set:
- **Novel signal** (genuinely new information like source spreads): expect **30-50% of pre-estimate**, floor 1pp
- **Proxied signal** (information the model can infer from price behavior): expect **10-20% of pre-estimate**, floor 0pp
- **Data quality improvements** (outlier voting, source reliability): **not subject to diminishing returns** — improves ALL existing features. BUT: if 99%+ of training data is already single-source backfill, the training impact is ~0pp. Impact concentrated at inference time (latest multi-source prices).

### Cumulative Ceiling

The combined improvement from completing ALL remaining work is likely **+5-8pp** (current 60-68% → 65-76%), not the +20-30pp that summing initial estimates would suggest.
