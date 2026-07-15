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

## High Impact (~3-8pp potential)

### 2. Supply-side features (collection rarity, wear tiers)

The model has no concept of supply constraints. Items from rare collections (Disco Volante, etc.) or high-wear tiers have fundamentally different supply dynamics than common skins. If the `items` table can be extended with `rarity`, `collection`, or `weapon_type`, these would be high-value features.

**Implementation:**
- Add columns to items table or parse from item name
- One-hot encode as categorical features
- Alternatively, collect float/pattern distribution data from third-party markets

**Effort:** Medium (needs schema change or name parsing)
**Impact estimate:** +3-6pp

---

### 3. Permutation-based feature validation + auto-prune (DONE ✅)

**Replaces:** the original SHAP-based elimination idea.

Built into `train()`: after CV, `_validate_feature_groups()` runs a fast permutation test on each feature group (e.g., `player_counts`, `price_technicals`, `events`) using the held-out validation set. Groups that drop less than 0.5pp when shuffled are flagged.

`prune_failed_groups=True` (default): when a group fails validation, its features are removed and the horizon is **retrained** without them. This prevents the spurious-accuracy-from-extra-capacity problem automatically.

**Configuration:** `ItemForecaster(db, prune_failed_groups=False)` to disable auto-prune during debugging.

---

## Moderate Impact (~1-4pp potential)

### 4. Event decay optimization

Decay constants are hardcoded: major=60d, operation=21d, case_drop=14d, update=7d, game_update=7d. These were domain-informed but never optimized.

**Implementation:**
- Grid search over tau values per event type on validation data
- Or add tau as a learnable parameter

**Effort:** Low (add grid search loop in `_add_event_features`)
**Impact estimate:** +1-2pp

**Already documented in:** `docs/2026-07-11-accuracy-improvement-brainstorm.md` (Item #7)

---

### 5. Multi-horizon consistent training

Currently each horizon (3d, 7d, 14d, 30d) is trained independently. A single model predicting all four horizon returns simultaneously would force shared representations of market dynamics and naturally enforce consistency (e.g., 3d direction ≤ 7d direction ≤ 30d direction).

**Implementation:**
- Multi-output regression target: `[target_return_3d, target_return_7d, target_return_14d, target_return_30d]`
- Sum of quantile losses across horizons as objective
- Or train a single model with horizon as a categorical feature

**Effort:** Medium (restructure target prep, model output, and inference)
**Impact estimate:** +2-4pp

**Already documented in:** `docs/2026-07-11-accuracy-improvement-brainstorm.md` (Item #13)

---

### 6. Listing count feature

Number of active Steam market listings at prediction time is a powerful short-term signal — items with few listings can spike on single buys, while oversupplied items face downward pressure.

**Implementation:**
- CSMarketAPI's `/v1/items` endpoint returns `listing_count`
- Collect during aggregator runs, store in a table
- Join as a daily feature

**Effort:** Medium (requires aggregator changes + new table)
**Impact estimate:** +3-8pp (speculative, depends on data quality)

---

## Deeper Architectural Changes (higher effort)

### 7. Item-type sub-models

Instead of one-hot encoding type as a feature, train separate LightGBM models per item category (skin model, sticker model, case model, etc.). Each model would specialize in its category's dynamics.

**Pros:**
- Captures category-specific feature interactions without global tree splits
- Each model can have its own hyperparameters

**Cons:**
- 5x model count (180 total instead of 36)
- Requires sufficient training data per category
- More complex deployment

**Effort:** High (significant refactor of training/prediction pipeline)
**Impact estimate:** +2-5pp

**Already documented in:** `docs/2026-07-11-accuracy-improvement-brainstorm.md` (Item #14)

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

---

### 9. Expanded training window (730d → 1460d+)

Currently training on 730 days of data. The parquet archive has data back to 2013. Longer training windows would capture more complete market cycles (multiple operation peaks, summer troughs, etc.), especially for the 30d horizon.

**Implementation:**
- Change `days_back=730` to `days_back=1460` in `train()` and `build_training_data()`
- May need to increase max_rows or add downsampling for old data

**Effort:** Low (single parameter change)
**Impact estimate:** +1-2pp (speculative — more data may not help if patterns have regime-shifted)

---

### 10. Optuna trial count increase

Currently 15 Optuna trials per quantile per horizon (60 total). Increasing to 30-50 trials would find better hyperparameters, especially for the larger search space.

**Effort:** Trivial (parameter change)
**Impact estimate:** +0.5-1pp (diminishing returns)

---

## Summary priority matrix

| # | Improvement | Effort | Impact | Data needed? | Already noted? | Status |
|---|-------------|--------|--------|-------------|----------------|--------|
| 1 | Player count | Low | +1-3pp | Collected | brainstorm #8 | **Done** — zero causal impact |
| 2 | Supply-side features | Medium | +3-6pp | Schema change | No | Pending |
| 3 | Permutation-based auto-prune | Low | Prevents overfit | No | **No** | **Done** |
| 4 | Event decay opt | Low | +1-2pp | No | brainstorm #7 | Pending |
| 5 | Multi-horizon | Medium | +2-4pp | No | brainstorm #13 | Pending |
| 6 | Listing count | Medium | +3-8pp | New collection | **No** | Pending |
| 7 | Sub-models | High | +2-5pp | No | brainstorm #14 | Pending |
| 8 | Conformal pred | Medium | Intervals only | No | brainstorm #12 | Pending |
| 9 | More training data | Low | +1-2pp | Collected | No | Pending |
| 10 | More HP trials | Trivial | +0.5-1pp | No | No | Pending |

**Top recommendation:** Start with **#2 (supply-side features)** — highest remaining impact opportunity.

**Guardrail:** Any new feature group must pass `_validate_feature_groups()` (built-in permutation test during `train()`) or it will be auto-pruned. This applies to all items above.
