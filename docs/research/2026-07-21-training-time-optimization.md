# Training Time Optimization: 64 min → ~14 min

> **RETIRED 2026-08-09. Do not optimise from this document — its premise is
> inverted.** It puts Optuna at ~38 min / **59%** and CV + calibration at
> ~1.5 min / **2%**. Measured on the shipped config
> (`docs/changelog/2026-08-09-shipped-retrain-cost-measured.md`): conformal CV is
> **439.3s / 50.4%** of an 872s retrain and Optuna is **35.0s / 4.0%**. The two
> phases traded places. Anyone optimising from the table below will chase Optuna
> and leave the actual bottleneck alone.
>
> Lever by lever:
>
> - ~~**E.** `LightGBMPruningCallback` + `HyperbandPruner`~~ — **refuted in code.**
>   The objective now runs fixed rounds with no pruner and no early stopping,
>   because the callback prunes on LightGBM's reported `quantile` metric rather
>   than on what the objective returns
>   (`backend/models/forecaster.py:2930-2934`).
> - ~~**G.** GOSS instead of bagging~~ — **shipped, then reverted.** The quantile
>   objective emits constant ±alpha gradients, so GOSS's gradient ranking is
>   degenerate and it shipped 1–2-tree q50 boosters. An A/B on 2026-07-29 restored
>   bagging (`_row_sampling_params`, `forecaster.py:2602-2635`);
>   `_ROW_SAMPLING_KEYS` (`:2599`) exists only to strip stale GOSS keys out of
>   cached params.
> - ~~**J.** 7d Optuna at 100 rounds~~ — **inverted.** The objective trains
>   `CV_FIXED_BOOST_ROUNDS` (`:619`), so 7d Optuna trains **500** rounds, and
>   Optuna is now **392.3s of a 1426.3s** cold retrain — the largest single phase
>   (`docs/changelog/2026-08-09-training-cost-levers.md`).
> - ~~**H.** Parallel ensemble training~~ — **premise gone.** `N_ENSEMBLES = 1`
>   (`:320`) and the p10/p90 models no longer exist, so "36 models" is 4 q50
>   boosters.
>
> **The arithmetic never closed either.** Stated savings sum to 31 min
> (10+8+3+5+4+1), and 64 − 31 = **33 min**, not the title's "~14 min"; the
> "After" table sums to **12.0 min** but is totalled at ~14; and the full/warm
> split double-counts rows already marked "skipped on warm retrain".
>
> **What survives:** the param values (47 / 0.01 / 0.0 / 1.5) did land, at
> `forecaster.py:2957-2963` (3d warm-start) and `:4104-4109` (fallback). Lever
> **F** was *not* taken as written — `SKIP_HP_HORIZONS` is `[3]` (`:337`), not
> `[3, 14, 30]`, deliberately.
>
> For current cost read `docs/architecture/model-optimization.md` → "Where the
> time goes now" and `docs/research/2026-08-09-model-and-data-research.md` §2.
> Every `file:line` below is stale; `forecaster.py` is now 6,719 lines.

**Date:** 2026-07-21
**Goal:** Aggressively shorten retrain time while minimizing accuracy loss (est. -0.3 to -1.1pp)

---

## Current Breakdown

| Phase | Time | % | Bottleneck |
|-------|:----:|:-:|------------|
| Optuna HP search | ~38 min | 59% | 3d=50 trials × 3 quantiles @ 200 rounds each; pruning is dead code (`trial.report()` at step=0 with `n_warmup_steps=5`) |
| Ensemble training | ~16 min | 25% | Sequential Python loop: 36 models (4 horizons × 3 quantiles × 3 members), up to 1000 rounds each |
| Regime models | ~5 min | 8% | bear/range/bull duplicates ensemble training |
| Data + features | ~3.5 min | 5% | DuckDB Parquet scan + rolling windows |
| CV + calibration | ~1.5 min | 2% | 6 expanding-window folds |
| **Total** | **~64 min** | 100% | |

---

## Changes (single file: `backend/models/forecaster.py`)

### E. Fix Optuna Pruning (lines 1636–1656)

**Bug:** `trial.report(score, 0)` at step=0 + MedianPruner with `n_warmup_steps=5` → pruning never fires.

**Fix:** Use `LightGBMPruningCallback` (auto-reports every iteration) + switch to `HyperbandPruner`.

```python
from optuna.integration import LightGBMPruningCallback

# Replace the objective's lgb.train block:
opt_callbacks = [
    lgb.log_evaluation(0),
    LightGBMPruningCallback(trial, "quantile"),
]
if boosting_type != "dart":
    opt_callbacks.insert(0, lgb.early_stopping(20))
model = lgb.train(params, dtrain, num_boost_round=200,
                  valid_sets=[dval], callbacks=opt_callbacks)
return model.best_score["valid_0"]["quantile"]

# Replace pruner:
pruner = optuna.pruners.HyperbandPruner(
    min_resource=5, max_resource=200, reduction_factor=3
)
```

**Save:** ~10 min. **Risk:** Low (official integration).

---

### F. Freeze 3d Params (lines 151, 2296–2302)

Add 3 to `SKIP_HP_HORIZONS`:
```python
SKIP_HP_HORIZONS = [3, 14, 30]  # was [14, 30]
```

Update fallback defaults to match the validated warm-start params from the 50-trial depth experiment:
```python
# Lines 2296-2302 replacement:
base_params["num_leaves"] = 47     # was 31
base_params["learning_rate"] = 0.01  # was 0.03
base_params["lambda_l1"] = 0.0      # was 0.5
base_params["lambda_l2"] = 1.5      # was 0.5
```

**Save:** ~8 min (3d was 50 trials × 3 quantiles). **Risk:** Low (warm-start validated).

---

### G. GOSS Instead of Bagging (lines 1631–1634, 2265–2268, 2288–2292)

Replace `bagging_fraction` + `bagging_freq` with `data_sample_strategy='goss'`.

**In Optuna params (line 1631-1634):**
```python
"min_gain_to_split": 0.1,
"feature_fraction": 0.7,
"data_sample_strategy": "goss",
"top_rate": 0.2,
"other_rate": 0.1,
```

**In base params (lines 2265-2268):** same swap.

**Remove from merge_keys (line 2290):** `"bagging_fraction"`.

**Remove from warm-start enqueue (line 1669):** `"bagging_fraction": 0.7` line.

**Save:** ~3 min. **Risk:** Medium (GOSS vs bagging accuracy unknown for this dataset).

---

### H. Parallel Ensemble Training (lines 2322–2328, 2391–2397)

Train 2 ensemble members concurrently with `n_jobs` scaled down per worker:

```python
from concurrent.futures import ThreadPoolExecutor
import os

n_workers = min(self.N_ENSEMBLES, max(1, (os.cpu_count() or 4) // 2))
cpu_per_worker = max(1, (os.cpu_count() or 4) // n_workers)

ensemble_models = []
with ThreadPoolExecutor(max_workers=n_workers) as pool:
    futures = []
    for ei in range(self.N_ENSEMBLES):
        p = pq.copy()
        p["random_state"] = self.ENSEMBLE_SEEDS[ei]
        p["feature_fraction"] = self.ENSEMBLE_FEATURE_FRACTIONS[ei]
        p["n_jobs"] = cpu_per_worker
        futures.append(pool.submit(self._train_ensemble_member, p, dtrain, dval, boost_rounds))
    for f in futures:
        ensemble_models.append(f.result())
```

Same pattern for regime ensemble (lines 2391–2397).

**Save:** ~5 min. **Risk:** Medium (thread safety — Dataset objects are read-only after creation).

---

### I. Skip Regime + Feature Val on Warm Retrain (lines 2353, ~2478)

```python
# Line 2353:
if os.environ.get("SKIP_REGIMES") == "1" or _warm_retrain:

# Before feature validation block (~2478):
if _warm_retrain:
    logger.info("  Skipping feature-group validation (warm retrain)")
elif len(val_set) < 2000 or val_dates < 7:
    ...
```

**Save:** ~4 min. **Risk:** Low.

---

### J. Reduce 7d Optuna Rounds (lines 1644–1648)

```python
_num_rounds = 100 if horizon == 7 else 200
model = lgb.train(params, dtrain, num_boost_round=_num_rounds, ...)
```

**Save:** ~1 min. **Risk:** Low (pruning already stops early).

---

## Time Budget (After All Changes)

| Phase | After | Notes |
|-------|:-----:|-------|
| Data loading | ~2 min | Fixed cost |
| Feature engineering | ~1.5 min | Fixed cost |
| Optuna (7d only, 15×3) | ~2 min | Hyperband prunes bad trials, 100 rounds |
| Ensemble (parallel 2-wide) | ~6 min | 36 models → 18 sequential groups × 2 parallel |
| Regime models | ~0.5 min | Skipped on warm retrain |
| CV + calibration | ~0 | Skipped on warm retrain |
| **Full retrain** | **~14 min** | First time or forced |
| **Warm retrain** | **~10 min** | Cached HP + skip regime/CV |

---

## Accuracy Risk

| Change | Est. DA Impact | Rationale |
|--------|:-------------:|-----------|
| E. Fix pruning | +0 to +0.3pp | Hyperband finds better params faster |
| F. Freeze 3d params | -0.1 to -0.3pp | May miss slightly better combos |
| G. GOSS | +0 to -0.5pp | LGBM paper: "almost same accuracy" |
| H. Parallel ensemble | 0pp | Identical training |
| I. Skip regime/val | -0.1 to -0.3pp | Rare regimes have limited impact |
| J. 7d rounds 100 | -0.1pp | Early stopping covers this |
| **Total** | **-0.3 to -1.1pp** | |

---

## Refs

- `backend/models/forecaster.py` — all changes in this file
- LightGBM GOSS: https://lightgbm.readthedocs.io/en/latest/Parameters.html#data_sample_strategy
- Optuna HyperbandPruner: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html
- Optuna LightGBMPruningCallback: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.LightGBMPruningCallback.html
