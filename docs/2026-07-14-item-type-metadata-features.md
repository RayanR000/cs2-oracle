# Item-Type Metadata Features — 2026-07-14

## Goal

Add item-type metadata (skin, case, sticker, graffiti, musickit) as categorical features so the model can learn category-specific price dynamics instead of treating all items interchangeably.

## Problem

The model had no concept of what type of item it was predicting. A $0.03 sticker and a $2000 knife received identical features. Different item categories have fundamentally different:
- Volatility profiles (stickers are volatile, knives are slow)
- Response to events (cases spike on rotation announcements, stickers spike during majors)
- Seasonality (cases follow operation cycles, skins follow collection supply)
- Trade velocity and price elasticity

Without type information, the model had to infer category-specific patterns purely from price/volume time series — requiring many tree splits and wasting capacity.

## Implementation

### New methods in `backend/models/forecaster.py`

**`_fetch_item_metadata(self) -> pd.DataFrame`** (line 294)
- Queries `SELECT item_id, type FROM items` from the database
- Caches result in `self._item_meta_cache` to avoid repeated queries
- Returns empty DataFrame on failure (graceful fallback for tests/mock DB)

**`_add_item_metadata_features(self, df: pd.DataFrame) -> pd.DataFrame`** (line 308)
- Merges item type onto the feature DataFrame on `item_id`
- One-hot encodes type into binary columns: `item_type_skin`, `item_type_sticker`, `item_type_case`, `item_type_graffiti`, `item_type_musickit`, `item_type_unknown`
- Drops the raw string `type` column after encoding

**Integration**
- Called inside `engineer_features()` (line 590), after temporal features and event features but before returning. This means it runs during both training (`build_training_data`) and prediction (`predict`), and is automatically included in walk-forward evaluation.

### Files modified

- `backend/models/forecaster.py` — added `_fetch_item_metadata()` at line 294, `_add_item_metadata_features()` at line 308, call in `engineer_features()` at line 590

## Results

Walk-forward evaluation on 50 items across 2022-2026 parquet archive (26 expanding windows, 60-day steps, fixed LGB params, no ensemble). Same methodology as `evaluate_forecaster.py`.

| Horizon | Before | After | Change |
|---------|:------:|:-----:|:------:|
| **3d**  | 61.1%  | 61.2% | +0.1pp |
| **7d**  | 60.7%  | 60.4% | -0.3pp |
| **14d** | 61.4%  | 61.7% | +0.3pp |
| **30d** | **64.4%** | **68.5%** | **+4.1pp** |

### Key observations

- **30d horizon gains +4.1pp** (64.4% → 68.5%) — item type is a longer-term signal, and category-specific dynamics compound over longer windows.
- **Fold std dev decreased** from 10.7% to 9.3% for 30d — predictions are more consistently accurate across time periods.
- **Short horizons (3d/7d/14d) within fold variance** — type information adds little to short-term momentum features (returns, RSI, Bollinger) that already capture most of the short-term signal.
- The `item_type_skin` and `item_type_sticker` features likely carry the most weight given they represent ~90% of items (20,903 skins + 11,632 stickers out of 35,058 total).

### Comparison to earlier accuracy stages

| Stage | 7d Dir Acc | 30d Dir Acc | Changes |
|-------|:----------:|:-----------:|---------|
| Pre-audit (MA-crossover) | ~34% | ~34% | Random baseline |
| After P1/P2 fixes | 70.9% | 72.5% | *(buggy targets)* |
| After target inversion fix | 61.1% | 65.8% | Genuine accuracy established |
| **After item-type features** | **60.4%** | **68.5%** | **+4.1pp on 30d** |

## Next Steps

- The type feature is a simple one-hot encoding. A richer representation (e.g., learned embeddings via category-wise feature interactions in LightGBM's categorical support) could yield more.
- ~~Item metadata could be extended with rarity (blue/purple/pink/red/gold), collection name, or weapon type — parsed from the `name` field or added to the `items` table schema.~~
- **✅ Done:** Rarity and weapon_type features implemented (2026-07-15). See `docs/changelog/2026-07-15-supply-side-features.md`. Impact: +0.66pp avg directional accuracy.
- **Outstanding:** Collection name parsing — requires `csmarketapi_reference.db` backfill which was never run.
