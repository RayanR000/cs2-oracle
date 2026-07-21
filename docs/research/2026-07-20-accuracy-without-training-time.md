# Accuracy Improvements with Minimal Training Time Overhead

> **Date:** 2026-07-20
> **Context:** Brainstorming improvements prioritized by impact per training-time-minute.
> Current production accuracy: **3d=61.5%, 7d=52.8%, 14d=55.7%, 30d=54.2%** (lgbm-v3).
> Full retrain: ~59 min (Optuna 38 min, ensemble training 16 min, data+features 4 min).

## Current Training Time Breakdown

| Phase | Time | % | Bottleneck |
|-------|------|---|------------|
| Optuna HP search (3d: 10 trials, 7d: 15 trials) | ~38 min | 64% | 6D search space, 200 rounds/trial |
| Ensemble training (4 horizons × 3 quantiles × 3 members = 36 models) | ~16 min | 27% | Sequential Python loop, up to 1000 rounds each |
| Feature engineering (~200-400K rows) | ~1.5 min | 2.5% | Rolling windows + cross-sectional groupbys |
| Data loading (DuckDB Parquet scan + multi-source voting) | ~2 min | 3% | 5.5M row archive scan |
| CV evaluation + calibration + pruning | ~1.5 min | 2.5% | Expanding-window folds |

---

## Tier 1: ~0 Added Training Time (Config Flags Only)

These are **already implemented but inactive.** Flipping config constants costs nothing and adds no training time — some even reduce it.

### 1.1 Enable DART Boosting for 14d / 30d

| | |
|---|---|
| **File:** | `backend/models/forecaster.py:97` |
| **Change:** | `BOOSTING_TYPE_MAP = {3: "gbdt", 7: "gbdt", 14: "gbdt", 30: "gbdt"}` → `{3: "gbdt", 7: "gbdt", 14: "dart", 30: "dart"}` |
| **Why inactive:** | Docs claim DART is active, code shows all 4 horizons hardcoded to `"gbdt"`. DART code path is complete: Optuna searches `drop_rate/max_drop/skip_drop` when `boosting_type == "dart"`, and `DART_NUM_BOOST_ROUND = 500` is half of GBDT's 1000. |
| **Training time:** | ~Same or **faster** (500 rounds vs 1000). |
| **Expected impact:** | **+0.5-2pp** on 14d/30d. DART reduces overfitting via tree dropout, which matters on noisy longer horizons where the model tends to predict "flat". |
| **Risk:** | DART can be less stable on small datasets. The 400K-row training set should be sufficient. |

### 1.2 Search `bagging_fraction` in Optuna

| | |
|---|---|
| **File:** | `backend/models/forecaster.py:1567` |
| **Change:** | Add `bagging_fraction` to Optuna's `suggest_float` alongside the 6 existing params. Currently fixed at 0.7. |
| **Training time:** | **None** — reuses existing Optuna trials that already run. |
| **Expected impact:** | **+0.3-1pp**. Bagging fraction controls row subsampling per iteration, which is a distinct regularization lever from column subsampling (`feature_fraction`). LightGBM defaults recommend tuning it together with `feature_fraction`. |
| **Integration:** | Single line: `params["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.3, 0.9, step=0.1)` + add `"bagging_fraction"` to the merge keys list (line 2139). |

### 1.3 Increase Forecast Blending Weight

| | |
|---|---|
| **File:** | `backend/models/forecaster.py` (class constant) |
| **Current:** | `FORECAST_BLEND_WEIGHT = 0.15` |
| **Proposed:** | Validate weights in `[0.1, 0.25]` via a quick walk-forward sweep. |
| **Training time:** | **None** — post-processing only. |
| **Expected impact:** | **+0.2-0.5pp** by reducing daily direction flip-flopping. |
| **Implementation:** | Add a one-time sweep script `scripts/tune_blend_weight.py` that iterates weights on held-out dates, measuring directional accuracy. |

### 1.4 Expand Optuna Search to 7 Horizon-Specific Params

| | |
|---|---|
| **Why:** | Currently `min_gain_to_split` (0.1) and `max_bin` (63) are never searched. `boosting_type` is hardcoded. |
| **Expected impact:** | **Negligible** alone — `max_bin` is a speed/accuracy tradeoff already optimized. `min_gain_to_split` matters at very low learning rates. |
| **Verdict:** | **Skip** — diminishing returns for the Optuna trial budget. |

---

## Tier 2: +1-5 min Training Time

### 2.1 Enable HP Search for 14d

| | |
|---|---|
| **File:** | `backend/models/forecaster.py:99` |
| **Current:** | `SKIP_HP_HORIZONS = [14, 30]` |
| **Change:** | `[30]` only (or empty list). |
| **Training time:** | **+1-2 min** (15 Optuna trials × ~5s each × 3 quantiles for 14d only). |
| **Expected impact:** | **+0.5-1pp**. 14d currently uses default params (depth=5, leaves=31, lr=0.03) which may be suboptimal for the 14d return distribution. |
| **Why 14d before 30d:** | 14d has marginally stronger signal (55.7% vs 54.2%) and the ensemble is more likely to benefit from tuned learning rate/regularization. |

### 2.2 Enable HP Search for 30d

| | |
|---|---|
| **Change:** | Remove 30 from `SKIP_HP_HORIZONS`. |
| **Training time:** | **+1-2 min**. |
| **Expected impact:** | **+0.3-1pp**. 30d is the noisiest horizon. Tuned regularization (higher L1/L2, lower learning rate) may reduce the "predict flat" bias that currently depresses DA. |

### 2.3 Increase 3d Optuna Trials: 10 → 15

| | |
|---|---|
| **File:** | `backend/models/forecaster.py:98` |
| **Current:** | `N_TRIALS_MAP = {3: 10, 7: 15}` |
| **Change:** | `{3: 15, 7: 15}` (with warm-start, the 3d converges by ~trial 8-10, but 5 more trials gives TPE room to explore better L2/depth combos). |
| **Training time:** | **+30s**. |
| **Expected impact:** | **+0.1-0.3pp** (diminishing returns — 3d already has warm-start from the 50-trial depth experiment). |

---

## Tier 3: +5-15 min Training Time

### 3.1 Expand Ensemble: 3 → 5 Members

| | |
|---|---|
| **File:** | `backend/models/forecaster.py:88-90` |
| **Current:** | `N_ENSEMBLES = 3`, seeds `[42, 73, 91]`, feature fractions `[0.6, 0.7, 0.8]` |
| **Proposed:** | `N_ENSEMBLES = 5`, add seeds `[13, 128]` with fractions `[0.5, 0.9]`. |
| **Training time:** | **+10 min** (2 extra models × 12 quantile×horizon groups × ~1 min per ensemble member with up to 1000 rounds + early stopping). |
| **Expected impact:** | **+0.5-1pp**. The marginal gain of the 4th member is ~0.3-0.5pp, 5th member ~0.1-0.3pp. Variance reduction from averaging more decorrelated models. |
| **Why not 6:** | 6th member adds only ~0.1pp. 5 captures most of the ensemble benefit at lower cost. |

### 3.2 Accelerate Existing Ensemble via Parallelism

| | |
|---|---|
| **Problem:** | 36 ensemble models train sequentially in a Python for-loop. Each model internally uses `n_jobs=-1` (all cores), but models cannot overlap. |
| **Change:** | Use `concurrent.futures.ThreadPoolExecutor` or `joblib.Parallel` to train multiple ensemble members simultaneously (e.g., 3 at a time). LightGBM Python bindings release the GIL during training. |
| **Training time:** | **-30-50%** on ensemble phase (saves ~5-8 of the 16 min). |
| **Expected impact:** | **Zero** on accuracy — pure speed improvement, which can be reinvested into more members. |
| **Complexity:** | Medium — needs careful thread-safety for dataset objects. LightGBM's `Dataset` is not fully thread-safe; each member needs its own `lgb.Dataset`. |

---

## Tier 4: New Feature Engineering (~0 Training Time Impact)

Feature engineering adds negligible training time (<30s each for 400K rows). These are genuinely new signals not proxied by existing features.

### 4.1 Quality Spread / Cross-Wear Features

| | |
|---|---|
| **Signal:** | Items traded in multiple wears (FN, MW, FT, WW, BS) have price spreads that signal market conditions. A tightening spread (FN approaching MW) suggests demand concentration at the top end. A widening spread suggests liquidity stress. |
| **Implementation:** | Parse item names for wear suffix, group by base name, compute `price_wear_spread_pct = (p_FN - p_BS) / p_BS`, `wear_premium_fn_vs_mw`, `wear_spread_change_7d`. Store as a daily cross-sectional feature. |
| **Expected impact:** | **+1-2pp** (per accuracy-opportunities.md — genuinely novel signal, not proxied by price technicals). |
| **Data source:** | Existing Parquet archive already contains per-wear prices. The grouping key is the item name without wear suffix. |
| **Code location:** | New method `_add_wear_spread_features()` in `forecaster.py`, called from `engineer_features()`. |

### 4.2 Post-Spike Mean Reversion Speed

| | |
|---|---|
| **Signal:** | After a >20% daily price spike, items revert at different speeds. Fast reverters (return to baseline within 3d) behave differently from slow reverters (stay elevated). This measures the "memory" of price shocks. |
| **Implementation:** | Compute `days_to_revert_50pct` after spike events: how many days until price retraces 50% of the spike gain. Encode as `spike_reversion_speed` (fast/medium/slow) and `spike_magnitude_30d`. |
| **Expected impact:** | **+0.5-1pp** (simple measure of mean-reversion tendency not captured by rolling stats alone). |
| **Data source:** | Self-contained — computed from `daily` DataFrame within `_compute_price_features()`. |

### 4.3 Tournament Event Anticipation Features

| | |
|---|---|
| **Problem:** | Current event features use exponential decay from event date — they capture reaction but not *anticipation*. Skins of teams/players in upcoming majors start moving 1-2 weeks before the event. |
| **Improvement:** | Add "upcoming event" features: `days_until_next_major`, `days_until_next_operation`, event type of next event. These look forward instead of backward. |
| **Expected impact:** | **+0.5-1pp** (captures pre-event hype that current decay features miss entirely). |
| **Risk:** | Look-ahead bias if not implemented correctly — must ensure `days_until` is computed from the *past* date, not the current date. The training pipeline already handles this correctly via per-date feature computation. |

### 4.4 Price Clustering / Round-Number Resistance

| | |
|---|---|
| **Signal:** | Psychological price levels ($10, $50, $100, $500) act as resistance/support. Items near these levels behave differently. |
| **Implementation:** | `distance_to_nearest_round_number = price % round_level` where round_level = 10^floor(log10(price)). Also `is_near_round_number` flag. |
| **Expected impact:** | **+0.2-0.5pp** (weak signal, but cheap to compute). |

---

## Implementation Plan

### Phase 1: Free Lunch (do first, measure impact)

1. **Enable DART for 14d/30d** — 3 lines, literal toggle
2. **Search bagging_fraction** — +1 line to Optuna search space
3. **Verify impact** via backtest after next retrain

**Time investment:** ~15 min to implement. **Retrain stays at ~59 min.**

### Phase 2: Low-Cost HP Expansion

4. **Remove 14 from SKIP_HP_HORIZONS** — 1 line
5. **Remove 30 from SKIP_HP_HORIZONS** — 1 line
6. **Bump 3d trials 10→15** — 1 line

**Time investment:** ~5 min. **Retrain: +2-4 min → ~63 min.**

### Phase 3: Ensemble Expansion + Parallelism

7. **Expand ensemble 3→5** — add seeds + fractions
8. **Parallelize ensemble training** — `ThreadPoolExecutor` wrapper

**Time investment:** ~30 min. **Retrain: +2 min (parallel saves ~6 min, 2 extra members add ~10 min). → ~65 min.**

### Phase 4: New Features

9. **Quality spread** — new `_add_wear_spread_features()`
10. **Post-spike reversion** — within `_compute_price_features()`
11. **Tournament anticipation** — upgrade event features

**Time investment:** ~1-2 hours. **Retrain: +~30s → ~66 min.**

---

## Summary: Accuracy vs Training Time Tradeoffs

| # | Change | Added Time | Total Time | Est. Impact | Impact/Min |
|---|--------|:----------:|:----------:|:-----------:|:----------:|
| 1 | DART for 14d/30d | ~0 min | 59 min | +0.5-2pp | **∞** |
| 2 | bagging_fraction search | ~0 min | 59 min | +0.3-1pp | **∞** |
| 3 | Forecast blend tuning | ~0 min | 59 min | +0.2-0.5pp | **∞** |
| 4 | HP search 14d | +1-2 min | 61 min | +0.5-1pp | **0.5pp/min** |
| 5 | HP search 30d | +1-2 min | 63 min | +0.3-1pp | 0.4pp/min |
| 6 | 3d trials 10→15 | +30s | 64 min | +0.1-0.3pp | 0.3pp/min |
| 7 | Ensemble 3→5 (parallel) | +2 min | 66 min | +0.5-1pp | 0.3pp/min |
| 8 | Quality spread features | ~0 min | 66 min | +1-2pp | **∞** |
| 9 | Post-spike reversion | ~0 min | 66 min | +0.5-1pp | **∞** |
| 10 | Tournament anticipation | ~0 min | 66 min | +0.5-1pp | **∞** |

**∞** = zero or negligible training time impact (feature engineering is ~30s total).

### Recommended Go-First Order

```
Phase 1 (free):    DART + bagging_fraction + blend tuning     → est. +1-3pp
Phase 2 (cheap):   14d/30d HP search + 3d trial bump          → est. +1-2.5pp
Phase 4 (features): quality spread + reversion + anticipation  → est. +2-4pp
Phase 3 (moderate): ensemble 3→5 with parallization            → est. +0.5-1pp
```

Combined ceiling: **~+4.5-10.5pp** (realistic: +4-7pp given diminishing returns).

---

## Items Explicitly Out of Scope (High Training Time Cost)

| Item | Training Time Impact | Why Skipped |
|------|--------------------:|-------------|
| Multi-horizon joint training | +300-400% | Single model for all 4 horizons requires restructuring targets and loss |
| Item-type sub-models | +500% | 5 separate model sets instead of 1 |
| CatBoost (again) | +200% | Tested and degraded accuracy by 18-20pp |
| 1460d → 2920d window | +100% | Doubles data loading + feature engineering. Marginal gain near zero |
| Neural forecasting (N-BEATS, TFT) | +500%+ | Requires GPU, infrastructure changes, hyperparameter tuning at scale |
| Supply depth (sell_listings) | N/A | Dropped 2026-07-16 — change/velocity variant requires 30d history or paid backfill |
