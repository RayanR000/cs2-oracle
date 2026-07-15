# Supply-Side Features (Rarity, Weapon Type, Cross-Sectional)

**Date:** 2026-07-15

## Summary

Added supply-side features (collection rarity, weapon type) to the forecaster by parsing Steam's `type` field from `market_catalog.db` and inferring weapon type from item names for the remaining items.

## Changes

### New files

- **`backend/models/steam_types.py`** — Parser for Steam's type field: extracts rarity and weapon_type from 137 unique type values covering 23,490 catalog items
- **`backend/scripts/backfill_supply_metadata.py`** — Reads `market_catalog.db`, parses every item's type field, infers weapon type from name/slug for non-catalog items, writes `price-archive/item-metadata.parquet` (8,691 rows, 109 KB)
- **`backend/scripts/ab_test_supply_side.py`** — A/B test harness: walk-forward evaluation with vs without supply-side features

### Modified files

- **`backend/database.py`** — Added `rarity`, `rarity_rank`, `weapon_type` columns to `Item` model with indexes
- **`backend/models/forecaster.py`** — Added `_fetch_supply_metadata()`, `_add_supply_side_features()`, `_add_weapon_type_cross_sectional_features()`, updated `engineer_features()`, `build_training_data()`, `predict()`, `_feature_group()`
- **`backend/tests/test_forecaster.py`** — Updated feature count range assertion from `[45,120]` to `[45,200]` (all 46 tests pass)

### Data assets

- **`price-archive/item-metadata.parquet`** — 8,691 items, 109 KB. Covers 100% of items: 4,395 from catalog with full rarity+weapon_type, 4,296 inferred from name/slug. Used as primary source by the forecaster (no DB dependency for inference).

## Coverage

| Source | Items | Rarity | Weapon Type |
|--------|-------|--------|-------------|
| Market catalog (type field) | 4,395 | ✅ Parsed from Steam type | ✅ Parsed from Steam type |
| Inferred (name/slug) | 4,296 | ❌ NULL (no rarity in name) | ✅ 22 weapon types |
| Missing | 0 | 0 | 0 |

## A/B Test Results

Walk-forward evaluation on 50 items, 26 expanding windows, ~27k samples per horizon.

| Horizon | Control | Treatment | Δ |
|---------|---------|-----------|---|
| 3d      | 59.5%   | **61.4%** | **+1.92pp** |
| 7d      | 61.1%   | 60.9%     | -0.16pp |
| 14d     | 59.5%   | **60.2%** | **+0.79pp** |
| 30d     | 68.0%   | 68.1%     | +0.08pp |
| **Avg** |         |           | **+0.66pp** |

**Verdict:** Modest improvement, worth keeping. Impact concentrated at short horizons (3d best). Below the 3-6pp estimate — existing features already capture much of the signal.

## Feature Count

- Control (without): 110 features
- Treatment (with): 154 features (+44 supply-side features: 11 rarity dummies + 22 weapon_type dummies + ~11 weapon-type cross-sectional features)

## Architecture

The forecaster loads metadata from Parquet (`price-archive/item-metadata.parquet`) with DB fallback. Supply-side features are added in two places:

1. **`_add_supply_side_features()`** — called in `engineer_features()`, adds identity features (rarity ordinal, rarity one-hot, weapon_type one-hot). Grouped under `item_identity` for permutation testing.
2. **`_add_weapon_type_cross_sectional_features()`** — called in `build_training_data()` and `predict()`, adds per-date weapon-type group returns and item-vs-weapon-type signals. Grouped under `supply_side`. Uses `wt_` prefix to avoid feature name collisions.

## Remaining

- **Collection name features** — deferred. Requires `csmarketapi_reference.db` backfill which was never run.
- **Supabase PostgreSQL migration** — new `rarity`/`weapon_type` columns not yet migrated. Non-critical: forecaster reads from Parquet.
