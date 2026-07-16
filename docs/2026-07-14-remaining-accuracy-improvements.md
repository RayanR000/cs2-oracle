# Remaining Accuracy Improvements

Current accuracy (post item-type features): 60-68% directional across horizons (~10-18pp above 50% baseline).

Below are the remaining opportunities, grouped by estimated impact and effort.

---



## Completed

### 1. Player count feature ✅

**Status:** Implemented and backfilled. See `docs/2026-07-15-player-count-backfill-and-ab-test.md`.

**Actual impact:** **Zero** — permutation test showed the +3pp A/B delta was spurious (extra model capacity, not causal signal). See `docs/2026-07-15-player-count-backfill-and-ab-test.md#permutation-test-causality-check`.

**Key files:**
- `scripts/backfill_player_counts_to_parquet.py` — historical DB → Parquet
- `collectors/player_counts.py` — ongoing collection
- `models/forecaster.py` — `_fetch_player_counts()`, `_add_player_count_features()`
- `.github/workflows/aggregator-update.yml` — daily Parquet append

---



### 2. Supply-side features (rarity, weapon_type, weapon-type cross-sectional) ✅

**Status:** Implemented and A/B tested. See `docs/changelog/2026-07-15-supply-side-features.md`.

**Actual impact:** **+0.66pp avg** directional accuracy (3d: +1.92pp, 7d: -0.16pp, 14d: +0.79pp, 30d: +0.08pp). Below the 3-6pp estimate because existing features already capture much of the signal. Impact concentrated at short horizons.

**Key files:**
- `models/steam_types.py` — Steam type field parser (rarity + weapon_type extraction)
- `scripts/backfill_supply_metadata.py` — backfill script (catalog → Parquet + DB)
- `models/forecaster.py` — `_fetch_supply_metadata()`, `_add_supply_side_features()`, `_add_weapon_type_cross_sectional_features()`
- `price-archive/item-metadata.parquet` — 8,691 items, 109 KB

---



### 3. Permutation-based feature validation + auto-prune (DONE ✅)

**Replaces:** the original SHAP-based elimination idea.

Built into `train()`: after CV, `_validate_feature_groups()` runs a fast permutation test on each feature group (e.g., `player_counts`, `price_technicals`, `events`) using the held-out validation set. Groups that drop less than 0.5pp when shuffled are flagged.

`prune_failed_groups=True` (default): when a group fails validation, its features are removed and the horizon is **retrained** without them. This prevents the spurious-accuracy-from-extra-capacity problem automatically.

**Configuration:** `ItemForecaster(db, prune_failed_groups=False)` to disable auto-prune during debugging.

---



### 4. Event decay optimization (DONE ✅)

**Status:** Implemented — coordinate-wise grid search over tau values (major=30-180, operation=7-60, case_drop=3-45, update=2-21, game_update=1-14), then validated via full walk-forward A/B.

**Actual impact:** **Zero** — the walk-forward test showed the "optimal" taus (operation: 21→45) degraded accuracy by -0.57pp mean. The domain-informed defaults were already close to optimal. See `scripts/optimize_event_decay.py` and `scripts/compare_event_decay.py`.

**Key changes:**
- `models/forecaster.py` — `decay_constants` refactored from hardcoded dict inside `_add_event_features()` to instance attribute `self.event_decay_constants`, enabling easy swapping
- `scripts/optimize_event_decay.py` — coordinate-wise tau grid search (fast single-split mode)
- `scripts/compare_event_decay.py` — walk-forward A/B comparison harness

**Costs incurred:** None (defaults unchanged, just added scripts + refactor).

---



## Moderate Impact (~1-3pp potential)

### 5. Multi-source outlier voting ✅

**Status:** Implemented and A/B tested. See `backend/models/forecaster.py` — `_apply_multi_source_voting()`.

**Why this is different from feature groups:** It's a data quality improvement, not a new feature dimension. Reducing noise in existing price features improves ALL downstream features (lagged prices, returns, rolling stats, Bollinger, RSI, MACD, volume features). This avoids the diminishing returns pattern that plagues new feature groups.

**Training impact: 0pp** on full historical dataset (1.4M rows). 99.6% of training data is single-source STEAMCOMMUNITY backfill, so voting only affects 0.4% of rows. Average price change on affected rows is $0.19 — insufficient to move LightGBM directional accuracy.

**Inference impact: essential.** 100% of items with live aggregator data have ≥3 sources. Voting changes the latest price by:
- >0.5% on **96.4%** of items
- >5% on **74.4%** of items
- Up to **$34.28** (max outlier correction)

This directly cleans the `current_price` used to convert return predictions into dollar forecasts. Without voting, stale/spoofed source readings propagate into every forecast's price level.

**Key files:**
- `models/forecaster.py` — `fetch_price_history()` loads per-source data; `_apply_multi_source_voting()` applies consensus

**Effort:** Low
**Net impact:** 0pp on training accuracy, critical for inference quality

---

### 6. Listing count / supply depth feature

Number of active Steam market listings at prediction time is a genuinely novel signal — current features have zero supply-side data after weapon_type removal. Items with few listings can spike on single buys, while oversupplied items face downward pressure.

**Implementation:**
- CSMarketAPI's `/v1/items` endpoint returns `listing_count`
- Collect during aggregator runs, store in a table
- Join as a daily feature

**Effort:** Medium (requires aggregator changes + new table)
**Impact estimate:** +1-3pp (calibrated: 30-50% of original +3-8pp estimate)

**Costs:** Adds ~1-2 new columns, negligible training impact. Requires ongoing API calls during aggregator runs.

---

### 7. Multi-horizon consistent training

Currently each horizon (3d, 7d, 14d, 30d) is trained independently. A single model predicting all four horizons simultaneously would force shared representations and naturally enforce consistency.

**Implementation:**
- Multi-output regression target: `[target_return_3d, target_return_7d, target_return_14d, target_return_30d]`
- Sum of quantile losses across horizons
- Or train a single model with horizon as a categorical feature

**Effort:** Medium (restructure target prep, model output, inference)
**Impact estimate:** +1-2pp (calibrated: 30-50% of original +2-4pp estimate)

**Costs:** Training time increases ~4x. Inference unchanged.

---



## Deeper Architectural Changes (higher effort)

### 8. Regime-switching models

Separate LightGBM models for bull/bear/range regimes (detected via existing `market_regime` feature as classifier). Each model specializes in its regime's dynamics.

**Effort:** Medium
**Impact estimate:** +2-4pp during volatile periods only; +1-2pp averaged across all time (calibrated: 30-50% of original +3-8pp)

**Costs:** 3x model count (108 total). Training time +200% on top of base ensemble.

---

### 9. Item-type sub-models

Instead of one-hot encoding type as a feature, train separate LightGBM models per item category (skin, sticker, case, etc.). Each model specializes in its category's dynamics.

**Pros:** Captures category-specific interactions, custom hyperparams per category.
**Cons:** 5x model count, needs sufficient data per category, complex deployment.

**Effort:** High
**Impact estimate:** +1-3pp (calibrated: 30-50% of original +2-5pp)

**Costs:** 180 total models. 5x training time and memory.

---



### 8. Conformal prediction

Replace the ad-hoc quantile monotonicity fix (`np.minimum(p10, p50)`, `np.maximum(p50, p90)`) with proper conformal prediction on the validation set. Gives distribution-free coverage guarantees and avoids distorting quantile identities.

**Implementation:**
- Compute nonconformity scores on validation set
- Calibrate prediction intervals at target coverage levels
- Replace `_enforce_quantile_monotonicity` with conformal intervals

**Effort:** Medium (new calibration logic, integration with existing pipeline)
**Impact:** Better calibrated intervals (not necessarily higher directional accuracy)

**Already documented in:** `docs/2026-07-11-accuracy-improvement-brainstorm.md` (Item #12)

**Costs:** Negligible — calibration runs once per training, adds milliseconds to inference.

---



### 9. Expanded training window (730d → 1460d+)

Currently training on 730 days of data. The parquet archive has data back to 2013. Longer training windows would capture more complete market cycles (multiple operation peaks, summer troughs, etc.), especially for the 30d horizon.

**Implementation:**
- Change `days_back=730` to `days_back=1460` in `train()` and `build_training_data()`
- May need to increase max_rows or add downsampling for old data

**Effort:** Low (single parameter change)
**Impact estimate:** +1-2pp (speculative — more data may not help if patterns have regime-shifted)

**Costs:** Training time increases ~2x (more rows to process). Memory proportional to row count. May need duckdb query performance tuning.

---



### 10. Optuna trial count increase

Currently 15 Optuna trials per quantile per horizon (60 total). Increasing to 30-50 trials would find better hyperparameters, especially for the larger search space.

**Effort:** Trivial (parameter change)
**Impact estimate:** +0.5-1pp (diminishing returns)

**Costs:** Training time increases proportionally (2-3x for 2-3x trials). Training already takes ~5-10 min per quantile×horizon.

---



## Cost summary: accuracy vs training time

| Improvement | Accuracy | Calibrated | Training time | Other costs |
|---|---|:---:|---|---|
| Player count | 0pp | **0pp** ✅ tested | +2-5% | API collection, Parquet storage |
| Supply-side | +0.66pp | **+0.66pp** actual | +5-10% | Backfill script, Parquet (109 KB) |
| Auto-prune | Prevents overfit | **Prevents overfit** | +10-20% | Validation after each horizon |
| Event decay | 0pp (reverted) | **0pp** ✅ tested | — | None (script-only) |
| Multi-source outlier voting | +2-4pp | **0pp train / essential inference** | None | Negligible |
| Multi-horizon | +2-4pp est. | **+1-2pp** | +300-400% | Structural refactor |
| Listing count / supply depth | +3-8pp est. | **+1-3pp** | +1-2% | New data collection pipeline |
| Regime-switching models | +3-8pp est. | **+1-2pp** avg | +200% | 3x model count |
| Quality spreads (cross-wear) | +2-4pp est. | **+1-2pp** | +1-2% | None |
| Sub-models | +2-5pp est. | **+1-3pp** | +500% | 5x model count, deployment complexity |
| Conformal pred | Intervals only | Intervals only | Negligible | New calibration logic |
| More training data | +1-2pp est. | **+0-1pp** | +100% | Memory, DuckDB tuning |
| More HP trials | +0.5-1pp est. | **+0.5-1pp** | +200-300% | None |

Training time is roughly linear in feature count, row count, and model count. Paying the time cost is worth it when the accuracy improvement is real (supply-side: +0.66pp for +5-10% time). Features that fail validation (like player counts) get auto-pruned, so their time cost is only paid during the first training run.

## Summary priority matrix

| # | Improvement | Effort | Impact | Calibrated | Training time penalty | Status |
|---|---|:---:|---:|:---:|---|---|
| 1 | Player count | Low | 0pp | **0pp** ✅ tested | +2-5% | **Done** |
| 2 | Supply-side features | Medium | +0.66pp | **+0.66pp** actual | +5-10% | **Done** |
| 3 | Auto-prune | Low | Prevents overfit | **Prevents overfit** | +10-20% | **Done** |
| 4 | Event decay opt | Low | 0pp | **0pp** ✅ tested | None | **Done** |
| 5 | Multi-source outlier voting | Low | +2-4pp | **0pp train / essential inference** | None | **Done** |
| 6 | Listing count / supply depth | Medium | +3-8pp | **+1-3pp** | +1-2% | 🛑 **Dropped (2026-07-16)** |
| 7 | Multi-horizon joint training | Medium | +2-4pp | **+1-2pp** | +300-400% | Pending |
| 8 | Regime-switching models | Medium | +3-8pp | **+1-2pp** avg | +200% | Pending |
| 9 | Quality spreads (cross-wear) | Medium | +2-4pp | **+1-2pp** | +1-2% | Pending |
| 10 | Sub-models (per-category) | High | +2-5pp | **+1-3pp** | +500% | Pending |
| 11 | Conformal prediction | Medium | Intervals only | Intervals only | Negligible | Pending |
| 12 | More training data (730d→1460d) | Low | +1-2pp | **+0-1pp** | +100% | Pending |
| 13 | More HP trials (15→50) | Trivial | +0.5-1pp | **+0.5-1pp** | +200-300% | Pending |

**Top recommendation (as of 2026-07-14):** **#6 (listing count / supply depth)** — highest remaining potential from genuinely novel signal. 🛑 **Subsequently DROPPED on 2026-07-16** — see `docs/research/accuracy-opportunities.md` §1 DECISION and `docs/changelog/2026-07-16-drop-supply-depth.md`. Top remaining work is now regime-switching models.

**Guardrail:** Any new feature group must pass `_validate_feature_groups()` (built-in permutation test during `train()`) or it will be auto-pruned. This applies to all items above. A/B test deltas without permutation confirmation should be treated as upper bounds, not guarantees.
