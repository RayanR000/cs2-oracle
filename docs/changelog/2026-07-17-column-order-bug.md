# Critical: Column-ordering bug in `fetch_price_history`

**Date:** 2026-07-17

All previously measured accuracy metrics (lgbm-v2, lgbm-v3, and the tier-1 speedup cold retrain) are **invalid**. Every model was trained on volume data masquerading as price data.

---

## Bug: DataFrame columns misaligned with SQL SELECT order

### Parquet path (`fetch_price_history`, lines 145–156)

The DuckDB SQL query selects 5 columns in this order:

```sql
SELECT item_slug, day, mean_price AS price, volume, source
```

The DataFrame constructor mapped them with wrong column labels:

```python
# BEFORE (buggy):
df = pd.DataFrame(rows, columns=["item_id", "timestamp", "source", "price", "volume"])

# What actually happened:
#   tuple[0] = item_slug         → "item_id"    ✓
#   tuple[1] = day               → "timestamp"  ✓
#   tuple[2] = mean_price (real) →  "source"    ✗  ← price written to source column
#   tuple[3] = volume            →  "price"     ✗  ← volume written to price column
#   tuple[4] = source (str)      →  "volume"    ✗  ← source string written to volume column
```

After `pd.to_numeric` and `dropna(subset=["price"])`:
- `df["price"]` contained **volume** values (always numeric, so nothing dropped)
- `df["volume"]` contained **source strings** (coerced to NaN)
- `df["source"]` contained **actual price** values

The multi-source voting function then voted on **volume** as if it were price, and all downstream features (lags, returns, rolling stats, RSI, MACD, Bollinger, targets) were computed on volume data.

### DB fallback path (lines 183–195)

Same class of bug — different column order:

```sql
SELECT item_id, date(timestamp) AS day, source, AVG(price) AS price, SUM(volume) AS volume
```

```python
# BEFORE (buggy):
df = pd.DataFrame(rows, columns=["item_id", "timestamp", "price", "volume", "source"])

# Actual mapping:
#   item_id              → "item_id"    ✓
#   day                  → "timestamp"  ✓
#   source (str)         → "price"      ✗  ← source string written to price
#   AVG(price) AS price  → "volume"     ✗  ← price written to volume
#   SUM(volume) AS volume→ "source"     ✗  ← volume written to source
```

### Root cause

The `source` column was added to the SQL SELECT when the DuckDB `UNION ALL BY NAME` schema-mismatch fix was introduced. It was appended at the **end** of the SELECT clause (`... volume, source`) but inserted into the **middle** of the DataFrame columns list (`["item_id", "timestamp", "source", "price", "volume"]`) instead of at the **end** (`["item_id", "timestamp", "price", "volume", "source"]`).

The comment above the line was also wrong — it claimed the order was `(item_slug, day, source, mean_price, volume)` when the actual SELECT order was `(item_slug, day, mean_price, volume, source)`.

### Fix applied

**Parquet path** (line 156):
```python
df = pd.DataFrame(rows, columns=["item_id", "timestamp", "price", "volume", "source"])
```

**DB path** (line 195):
```python
df = pd.DataFrame(rows, columns=["item_id", "timestamp", "source", "price", "volume"])
```

Also fixed the misleading comment.

---

## Additional bugs discovered during the same session

### 1. Feature-count mismatch between training and predict (CRITICAL)

`self.feature_cols` is a single global list that gets **mutated in place** during training as each horizon prunes non-causal feature groups:

| Horizon | Features after pruning |
|---------|----------------------|
| 3d      | 133 → 86 |
| 7d      | 86 → 66 |
| 14d     | 66 (no further pruning) |
| 30d     | 66 (no further pruning) |

The final value (66) was saved to `meta.json` and used for predict. But the **3d model was trained with 86 features**, so `m.predict(X_batch)` crashed with:

```
The number of features in data (66) is not the same as it was in training data (86)
```

**Fix:** During `load_models()`, extract per-horizon feature names from each model's internal `model.feature_name()` (the LightGBM Booster stores the exact feature set it was trained with). During predict, build the full feature matrix using the **union** of all per-horizon feature sets, then subset to horizon-specific columns before each model call.

Files affected:
- `load_models()` — build `self.horizon_feature_cols` from `model.feature_name()`
- `predict()` — use `self.horizon_feature_cols[horizon]` to subset `X_batch` per horizon

### 2. Prior-forecast slug/ID type mismatch

`_fetch_prior_forecasts()` tried `int(iid)` on string slugs from Parquet (`'Medium Rare' Crasswater | Guerrilla Warfare`). The DB stores integer item IDs, but the predict path works with string slugs.

**Fix:** Build a `slug_to_id` mapping from the `items` table and use it to convert Parquet slugs to DB integer IDs before the forecast-blend lookup.

### 3. Ensemble training scoped inside `if/else` HP-reuse guard

The ensemble training loop was inside only the `else` (non-Optuna) branch. Under HP caching (normal warm retrains), only the last quantile (q=0.9) received ensemble training. The `if` (Optuna) branch also had the same structure.

**Fix:** Moved the ensemble training loop outside the `if/else` block so all 3 quantiles always get the full N_ENSEMBLES training.

### 4. Multi-source voting performance bottleneck

The `apply(vote)` function ran on all 6M rows, including ~6M single-source rows (where voting is a no-op). This caused multi-minute hangs.

**Fix:** Split into single-source (fast `groupby.agg`, ~6M rows) and multi-source (existing `apply(vote)`, ~200K rows) paths.

### 5. Rolling percentile feature using slow Python loop

`rolling(365).apply(lambda x: ...)` called a Python function for every rolling window — very slow.

**Fix:** Replaced with `rolling(365).rank(pct=True)` — Cython-vectorized, ~100x faster.

### 6. DuckDB schema mismatch for old Parquet files

Pre-2026 Parquet files lack the `source` column. The glob `price-*.parquet` union failed with schema mismatch.

**Fix:** Per-file schema detection: if `source` not in columns, add `NULL::VARCHAR AS source`.

### 7. Feature importance crash after pruning

`_get_feature_importance` referenced `self.feature_cols`, which changes between horizons due to pruning. When the model's feature count differed from `self.feature_cols`, it crashed.

**Fix:** Use `model.feature_name()` instead of `self.feature_cols` for feature importance extraction.

---

## Impact assessment

| Bug | Severity | Affected versions | Discovered |
|-----|----------|-------------------|------------|
| Column ordering (`price` ↔ `volume`) | **CRITICAL** | All (lgbm-v1 through lgbm-v3) | 2026-07-17 predict run |
| Feature count mismatch at predict | CRITICAL | All versions with per-horizon pruning | 2026-07-17 predict run |
| Prior-forecast slug/ID type | HIGH | All with forecast blending | 2026-07-17 predict run |
| Ensemble training scope | HIGH | lgbm-v3 tier-1 speedups only | 2026-07-17 audit |
| Multi-source voting perf | MEDIUM | All | 2026-07-17 audit |
| Rolling percentile perf | MEDIUM | All | 2026-07-17 audit |
| DuckDB schema mismatch | MEDIUM | Code with UNION ALL path | 2026-07-17 audit |
| Feature importance crash | LOW | lgbm-v3 with pruning | 2026-07-17 audit |

### What needs to happen next

- **Retrain** with all fixes applied to get genuine price-based models and valid accuracy metrics
- Then run **predict** and **backtest** to establish a real baseline

---

## Files changed

| File | Changes |
|------|---------|
| `models/forecaster.py` | Column ordering fix (lines 156, 195); per-horizon feature cols in load_models/predict; slug→ID mapping in _fetch_prior_forecasts; (previously: ensemble training fix, voting speedup, rolling percentile, schema mismatch, feature importance) |
