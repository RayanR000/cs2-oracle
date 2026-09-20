# Optimization & ML Promotion Design

**Date:** 2026-09-19
**Goal:** Optimize data/test/ML pipelines and promote working ML heads to first-class status. Portfolio-impressive side project — keep architectural sophistication, optimize engineering quality.

---

## Tier 1: ML Promotion + Quick Wins

### 1a. Exceedance Head → Default ON

**Current:** Gated behind `EXCEEDANCE_SCALE=1` (line 1565) and `EXCEEDANCE_HEAD=1` (line 1590). Both default OFF.
**Change:** Flip both `_enabled()` methods to default `True`: `os.environ.get("FLAG", "1") == "1"`.
**Files:** `forecaster.py` (2 method bodies)
**Constraint:** Retrain required in same PR (CLAUDE.md rule).

### 1b. Anomaly Classifier → Default ON + Isotonic Calibration

**Current:** Gated by `ANOMALY_GBM=1` (line 1712). Binary LightGBM, served as raw `anomaly_p`. AUC lift +0.128-0.153.
**Change:** Flip default ON. Add isotonic calibration (same pattern as exceedance head: `IsotonicRegression(out_of_bounds='clip').fit(oof_probs, oof_labels)`). Store calibrator in model artifacts.
**Files:** `forecaster.py` (flag method + training block at ~6508 + predict block at ~7819)

### 2. `skip_unused_groups=True` in Predict

**Current:** Training (line 5697) passes the flag. Predict (line 9738) does not. Wastes ~20-40% of predict FE time computing shelved feature groups.
**Change:** Add `skip_unused_groups=True` to the predict `engineer_features` call.
**Files:** `forecaster.py` (1 call site)
**Verify:** No code between `engineer_features` return and column pruning reads a shelved feature.

### 3. Default `SKIP_REGIMES=1`

**Current:** `os.environ.get("SKIP_REGIMES") == "1"` (line 6569). Defaults OFF → 12 extra LightGBM fits.
**Change:** Flip to `os.environ.get("SKIP_REGIMES", "1") != "0"`. Keep all regime code for portfolio value.
**Files:** `forecaster.py` (1 gate)
**Note:** predict() at line 9970 prefers regime models when artifacts exist. Must handle gracefully when no regime artifacts are present (already does — falls back to global).

### 4. Mark 3 Unmarked Slow Tests + Recover Fast Tests

**Recover:** Remove `pytestmark = pytest.mark.slow` from `test_forecaster.py` line 22. Add `@pytest.mark.slow` only to the ~4 test functions/classes that call `lgb.train()` (~lines 487, 535, 600, 2249).
**Mark slow:** Add `pytestmark = pytest.mark.slow` to `test_ab_harness_trainer.py`, `test_scale_model.py`, `test_ngboost_wiring.py`.
**Files:** 4 test files

### 5. `addopts` Default

**Change:** Add `addopts = "-m 'not slow'"` to `pyproject.toml` `[tool.pytest.ini_options]`.
**Files:** `pyproject.toml`

---

## Tier 2: Performance Optimization

### 6. Vectorize `_conformal_records`

**Current:** Python for-loop at lines 8491-8507, building dicts row by row. Runs 32 times.
**Change:** Build DataFrame directly from numpy arrays using boolean mask indexing.
**Files:** `forecaster.py`

### 7. Precompute Sample Weights

**Current:** `groupby.transform(lambda x: x.pct_change().rolling(30, min_periods=5).std())` at lines 6027-6030. Called ~36 times.
**Change:** Compute once in `prepare_targets`, store as column `_vol_weight`, slice per fold.
**Files:** `forecaster.py`

### 8. Cache `present_columns` in archive.py

**Current:** `DESCRIBE SELECT * FROM read_parquet([61 files], union_by_name=true)` on every call.
**Change:** `@functools.lru_cache` keyed on `(archive_dir, frozenset(file_list))`. Module-level.
**Files:** `db/archive.py`

### 9. SQL DISTINCT ON for Fallback Price Lookup

**Current:** `.all()` + Python first-per-item at lines 624-682. Fetches ~100K rows when ~1K needed.
**Change:** Raw SQL with `DISTINCT ON (item_id)` for Postgres. SQLite fallback: subquery with `ROW_NUMBER() OVER (PARTITION BY item_id ORDER BY timestamp DESC) = 1`.
**Files:** `collectors/pipeline.py`

### 10. Skip Shelved Dollar-Scale Features

Already handled by item 2 (`skip_unused_groups=True` in predict). Training already skips them. No separate change needed.

---

## Tier 3: Test Infrastructure

### 11. Create `conftest.py`

**Fixtures:**
- `mock_forecaster(tmp_path)`: `ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))` — opt-in, not autouse.
- `db_session()`: in-memory SQLite engine + sessionmaker + `Base.metadata.create_all`. Yields session, rolls back on teardown.
**Files:** New `backend/tests/conftest.py`; update consuming files incrementally (start with 5-10 highest-duplication files).

---

## Tier 4: New ML Features

### 12. Adaptive Conformal Calibration (ACI)

**Motivation:** Scalar q_hat over-covers at 87-92% vs 80% target. Per-date coverage swings 58-99%. Sigma-conditioning works (62→95% across deciles) but the served/OOF basis mismatch killed the fitted exponent.

**Design:** Adaptive Conformal Inference per Gibbs & Candès (2021).
- **Strata:** `(horizon, sigma_decile)` — 4 × 10 = 40 cells
- **Update rule:** `q_hat[s,t+1] = q_hat[s,t] + η(α - err[s,t])` where `err` is miscoverage indicator
- **Initialize:** From OOF conformal CV q_hat per stratum
- **Daily update:** From `forecast_outcomes` table via served_recalibration.py
- **Storage:** Extend `served_recalibration.py` factors dict from `{horizon: float}` to `{(horizon, sigma_decile): float}`
- **Serving:** `feedback_factor` in `candidate_predictions.py` becomes stratum-aware
- **Guard rails:** Clamp `[0.5, 2.0]`, minimum 5 observations per stratum before updating, conservative η=0.05
- **Fallback:** If stratum has insufficient data, use horizon-level factor (current behavior)

**Files:** `conformal.py` (new `calibrate_adaptive`), `served_recalibration.py` (extend), `forecaster.py` (wire in), `candidate_predictions.py` (stratum-aware factor)

### 13. Learned Source Quality Weights

**Motivation:** Vote weighting is rule-based; aggregator_sync degrades vote by ~17% on average.

**Design:**
- Train small GBM predicting per-source absolute error vs. next-day multi-source consensus
- Features: `(source, log_price, n_sources_present, day_of_week, source_staleness_rate)`
- Train on archive dates with ≥3 sources (multi-source era, ~87 dates but dense per-item)
- Output: predicted error magnitude per (source, item) → inverse as vote weight
- Integration: Replace equal weighting in `_apply_multi_source_voting` with learned weights
- Fallback: If model unavailable, fall back to current equal-weight voting

**Files:** New `models/source_weight_model.py`, `forecaster.py` voting section (~line 2487)

### 14. Data Quality Anomaly Detection

**Motivation:** 18-33% frozen quotes in single-source era. Bit-identical detection is too simple.

**Design:**
- Isolation Forest on per-date cross-sectional features:
  - `pct_items_unchanged`: fraction of items with identical price to prior day
  - `mean_abs_return`: average |daily return| across items
  - `source_count`: number of active sources on this date
  - `cross_item_correlation`: pairwise return correlation (high = feed artifact)
- Train on 2025 multi-source era as "normal"
- Output: `data_quality_score ∈ [0, 1]` per date
- Integration: Multiply into vote weight model (#13) as a source reliability discount
- API: Expose as diagnostic field on forecast response

**Files:** New `models/data_quality.py`, integration in forecaster.py

---

## Implementation Order

1. **Test infrastructure** (items 4, 5, 11) — no code deps, unblocks everything
2. **ML flag flips + predict optimization** (items 1a, 1b, 2, 3) — forecaster.py changes
3. **Vectorizations** (items 6, 7) — forecaster.py perf
4. **Data pipeline** (items 8, 9) — archive.py + pipeline.py
5. **Adaptive conformal** (item 12) — builds on existing conformal.py
6. **Source weights + data quality** (items 13, 14) — builds on voting code
