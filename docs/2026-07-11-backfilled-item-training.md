# Backfilled-Item Training Pipeline

## Motivation

The parquet archive (`price-archive/prices-*.parquet`) contains 9.9M rows across 8,691 items, but only **5,542 items** have deep historical data from the STEAMCOMMUNITY source (2013–2026). The remaining ~3,149 items have only a few days of aggregator snapshots — insufficient for meaningful rolling features, RSI, MACD, or 7d/30d return targets.

Prior to this change, the forecaster trained on **all 8,691 items** in the parquet, which included ~3,000 items with no usable history flooding the feature matrix with NaNs and wasting training capacity.

**Fix:** Filter training (and prediction) to only items that have STEAMCOMMUNITY source rows — the ~5,500 items with years of reliable OHLCV data.

## Changes Made

### 1. `backend/models/forecaster.py` — `fetch_price_history()`

Added `backfilled_only: bool = False` parameter. When `True`, the DuckDB query is filtered to only include `item_slug` values that have at least one row with `source = 'STEAMCOMMUNITY'`:

```python
slug_filter = ""
if backfilled_only:
    slug_filter = """
        AND item_slug IN (
            SELECT DISTINCT item_slug
            FROM read_parquet('{}/prices-*.parquet')
            WHERE source = 'STEAMCOMMUNITY'
        )
    """.format(archive_dir)
```

This fetches **all sources** for those items (not just STEAMCOMMUNITY), so the model gets both the deep backfilled history and recent aggregator snapshots for the same items.

### 2. `backend/models/forecaster.py` — `build_training_data()`

Added `backfilled_only` passthrough to `fetch_price_history()`.

### 3. `backend/models/forecaster.py` — `train()`

Changed `build_training_data(days_back=365)` → `build_training_data(days_back=365, backfilled_only=True)`.

### 4. `backend/models/forecaster.py` — `predict()`

Changed `fetch_price_history(days_back=365)` → `fetch_price_history(days_back=365, backfilled_only=True)`.

### 5. `backend/scripts/forecast_prices.py` — slug→ID mapping

The parquet stores item identifiers as string slugs (Steam market hash names, e.g. `"AK-47 | Redline (Field-Tested)"`), but the database uses integer `item_id` foreign keys. When writing forecasts to the `item_forecasts` table, slugs must be mapped to integer IDs via the `items.name` column.

Added slug→ID mapping before forecast writes:

```python
slug_rows = db.execute(
    text("SELECT id, name FROM items WHERE is_backfilled = 1")
).fetchall()
slug_to_id = {r.name: r.id for r in slug_rows}
```

Items that exist in the parquet but not in the DB (17 items — old/renamed Steam items) are skipped with a warning.

### 6. GitHub Workflows — data-archive checkout

The `price-archive/` directory is only on the `data-archive` branch, not the default branch. All 5 workflows that read parquet files now checkout the `data-archive` branch into an `archive/` subdirectory and create a symlink (`price-archive -> archive/price-archive`) so all existing `Path(__file__).parent.parent.parent / "price-archive"` references resolve correctly.

Workflows updated:
- `price-forecast.yml` — runs `forecast_prices.py`
- `backtest-accuracy.yml` — runs `backtest_accuracy.py` (historical mode)
- `daily-trend-analysis.yml` — runs `run_task.py trends` → `analyze_trends.py`
- `long-term-trend-analysis.yml` — runs `long_term_trend_analyzer.py`
- `event-correlation-analysis.yml` — runs `event_analyzer.py`

The `aggregator-update.yml` already had the dual checkout (it both reads and writes the archive). The `discover-new-items.yml` does not read parquet and was left unchanged.

## Items in Production DB

The Supabase `items` table contains **5,525 items** (all backfilled). The 31,908-item catalog from `build_market_catalog.py` is in a separate SQLite database and has never been imported into Supabase. All production queries filter through `is_backfilled = 1`.

## Training Results

| Metric | Value |
|--------|-------|
| Training data | 1,347,507 rows, 5,542 items |
| Feature matrix | 1,324,775 rows, 65 features (after pruning 27 correlated >0.95) |
| Train/val split (7d) | 200,000 train / 66,218 val (last 21 calendar days) |
| Train/val split (30d) | 200,000 train / 66,168 val |
| Training time (M4) | ~9.5 min (full: Optuna 30 trials + 18 ensemble models) |
| Optuna best (7d) | le=63, lr=0.079, λ1=1.5, λ2=0.5, d=8, min=15 |
| Optuna best (30d) | le=63, lr=0.079, λ1=1.5, λ2=0.0, d=8, min=15 |
| Confidence calibration | Binary high/low at 99.7% accuracy for "high" bucket |
| Top features (7d) | `log_return_7d`, `price_std_30d`, `price_accel_7d`, `volume_mean_30d` |
| Top features (30d) | `item_return_vs_market_30d`, `market_return_30d`, `macd_signal`, `macd_line` |

## Adding New Items (Future)

New items (from the 31,908 catalog not yet tracked) should not be added to the training set until they have **≥90 days of continuous aggregator data**:

| Data available | Can compute |
|---------------|-------------|
| ≥14 days | Lag features (1d, 3d, 7d), basic rolling (7d) |
| ≥30 days | All rolling windows (30d), RSI, MACD, Bollinger |
| ≥90 days | `target_return_7d` (need 90 + 7 = 97 days total) |
| ≥120 days | `target_return_30d` (need 120 + 30 = 150 days total) |

### Process to add new items:

1. Ensure the aggregator pipeline collects price snapshots for the new items
2. Wait until items have ≥90 days of daily data in the parquet (aggregator source)
3. Option A: Backfill them into STEAMCOMMUNITY source (if old data exists from Steam API)
4. Option B: Modify the `backfilled_only` filter in `fetch_price_history()` to include items with `COUNT(DISTINCT day) >= 90` instead of filtering on source
5. Retrain the model — it should pick them up automatically

## Files Modified

| File | Change |
|------|--------|
| `backend/models/forecaster.py` | Added `backfilled_only` to `fetch_price_history`, `build_training_data`, `train`, `predict` |
| `backend/scripts/forecast_prices.py` | Added slug→ID mapping via `items.name` column |
| `backend/models/saved_models/` | Cleaned up old single-model files (pre-ensemble format) |
| `.github/workflows/price-forecast.yml` | Added data-archive checkout + symlink |
| `.github/workflows/backtest-accuracy.yml` | Added data-archive checkout + symlink |
| `.github/workflows/daily-trend-analysis.yml` | Added data-archive checkout + symlink |
| `.github/workflows/long-term-trend-analysis.yml` | Added data-archive checkout + symlink |
| `.github/workflows/event-correlation-analysis.yml` | Added data-archive checkout + symlink |
| `docs/2026-07-11-backfilled-item-training.md` | This file |
