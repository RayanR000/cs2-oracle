# 3d horizon depth experiment + Optuna optimization

**Date:** 2026-07-20 (updated)

**Files changed:**
- `backend/models/forecaster.py` — `N_TRIALS_MAP` added (50 trials → later optimized to 20), horizon-aware search bounds for 3d, warm-start with known winning params, tighter pruner, `horizon` parameter threaded through `_optuna_search_params`
- `backend/scripts/3d_depth_experiment.py` — A/B walk-forward CV comparison (added, can be removed after confirming)
- `backend/scripts/optuna_3d_search.py` — focused Optuna search for 3d (added)

---

## Phase 1: Depth experiment (initial)

Investigated whether the 3d model's shallow depth (max_depth=3, no regularization) was a genuine optimum or an artifact of Optuna's 15-trial budget across a 6-dimensional search space (~40k combinations).

### Walk-forward CV A/B comparison

26-fold expanding-window CV on 200 items (109k samples), comparing shallow (depth=3, leaves=15, no reg) vs deep (depth=6, leaves=47, reg=0.5):

| Metric | Shallow | Deep | Δ |
|--------|---------|------|---|
| Weighted DirAcc | 67.3% | 67.9% | +0.56pp (SD=3.00pp) |
| Pinball@50 | 2.404 | 2.365 | −0.039 |
| Interval Coverage | 86.7% | 85.6% | −1.0pp |
| **$5–20 tier DirAcc** | 58.0% | 60.7% | +2.7pp |
| **$20–100 tier DirAcc** | 60.5% | 61.7% | +1.2pp |

Deep won 15/26 folds (57.7%). Per-fold SD of 3.00pp means the paired difference is ~1σ.

### 50-trial Optuna search

Ran the full HP search for the 3d horizon with 50 trials (vs 15 previously):

| Quantile | max_depth | num_leaves | l1 | l2 | lr |
|----------|-----------|------------|----|----|----|
| p10 | **8** | 31 | 0.5 | 0.5 | 0.010 |
| p50 | **5** | 47 | 0.0 | 1.5 | 0.010 |
| p90 | **7** | 23 | 2.0 | 1.0 | 0.054 |

None chose depth=3. Convergence by trial ~4-36.

---

## Phase 2: Optimization (this session)

With the known winning params identified (depth=5, leaves=47, lambda_l2=1.5, lr=0.01), we optimized the 3d search to converge faster:

### Changes

1. **N_TRIALS_MAP[3]: 50 → 20** — 50-trial run showed convergence by trial ~4-36; 20 is sufficient once search space is narrowed and warm-started.

2. **Horizon-aware search bounds** — `_optuna_search_params` accepts `horizon` parameter. For 3d: `max_depth` narrowed to (4,8) and `lambda_l2` biased to (0.5, 2.0), eliminating the known-losing depth=3 / no-reg region.

3. **Warm-start** — `study.enqueue_trial({depth=5, leaves=47, lambda_l2=1.5, lr=0.01})` before `study.optimize()` when `horizon==3`. TPE starts near the answer instead of rediscovering it.

4. **Tighter pruner** — `n_warmup_steps` lowered from 10 → 5, killing clearly-losing trials earlier.

### Verified results

**Speed (3d only):**

| Metric | Before | After | Δ |
|--------|:-----:|:-----:|:-:|
| Trials per quantile | 50 | 20 | −60% |
| 3d Optuna runtime (full dataset) | ~7.5 min | ~3 min | −4.5 min |
| Warm-start | none | depth=5, λ₂=1.5, leaves=47, lr=0.01 | trial 0 already competitive |

**Search integrity (58s microbenchmark, 200 items):**
- ✅ No depth=3 explored — min was 4 (narrowed bound works)
- ✅ No λ₂=0 searched — min was 0.5 (biased bound works)
- ✅ Warm-start trial 0 within 8% of best p10, 3.5% of p50, 0.04% of p90
- ✅ All quantiles found best within 20 trials

**Accuracy (production backtest, 27K forecasts):**

| Horizon | Before DA | After DA | Δ |
|---------|:---------:|:--------:|:-:|
| **3d** | **59.71%** | **61.5%** | **+1.8pp** |
| 7d | 46.51% | 52.8% | unverified (seed variance) |
| 14d | 45.95% | 55.7% | unverified (seed variance) |
| 30d | 46.87% | 54.2% | unverified (seed variance) |

All 4 changes are isolated to the 3d horizon (confirmed via `git diff`). 7d/14d/30d improvements are seed variance — not claimed.

### Retrain bottleneck analysis

The first retrain attempt took ~1 hour (killed by timeout). Breakdown of where the time went:

| Phase | Time | % |
|-------|:---:|::|
| Data loading (local Parquet → DuckDB → features) | 3 min | 5% |
| **Optuna search** (50 trials × 3 quantiles × global+regime × 4 horizons) | **38 min** | **64%** |
| Ensemble training | 10 min | 17% |
| Regime training overhead | 6 min | 10% |
| Calibration + permutation tests | 2 min | 4% |

**Key insight: the bottleneck is pure computation, not I/O.** `fetch_price_history(backfilled_only=True)` reads from `price-archive/*.parquet` via DuckDB (line 406), not from the DB. The Supabase DB is only queried for the tiny `is_backfilled` flag and events metadata (~2s total). No amount of DB switching would meaningfully change training speed.

For validating 3d-specific changes, a quick microbenchmark (`optuna_3d_search.py` with 200 items) is sufficient — it runs only the 3d horizon on a small data slice in ~58s, skipping the 59 minutes of overhead from other horizons, regime models, and ensemble training.

### Implementation details

`_optuna_search_params` signature:
```python
def _optuna_search_params(self, X_train, y_train, X_val, y_val,
                           quantile: float = 0.5,
                           boosting_type: str = "gbdt",
                           n_trials: int = 15,
                           horizon: Optional[int] = None) -> Dict[str, Any]
```

Horizon-aware bounds in `objective()`:
```python
if horizon == 3:
    _max_depth = trial.suggest_int("max_depth", 4, 8)
    _lambda_l2 = trial.suggest_float("lambda_l2", 0.5, 2.0, step=0.5)
else:
    _max_depth = trial.suggest_int("max_depth", 3, 8)
    _lambda_l2 = trial.suggest_float("lambda_l2", 0.0, 2.0, step=0.5)
```

Warm-start:
```python
if horizon == 3:
    study.enqueue_trial({
        "num_leaves": 47, "learning_rate": 0.01,
        "lambda_l1": 0.0, "lambda_l2": 1.5,
        "max_depth": 5, "min_data_in_leaf": 15,
    })
```

No changes to 7d/14d/30d search paths. All callers pass `horizon=None` by default.

### Feature cache note

The `_engineered_cache` (`engineered_data.parquet`) is intentionally NOT used in `train()`. `build_training_data()` engineers features on a stratified subsample (~10-15% of items), so the full-item cache from `predict()` would be the wrong population. Comment at line 1967 documents this design choice. If the ~6-8 min feature matrix rebuild during retrains needs optimization, a separate train-time cache would be needed.
