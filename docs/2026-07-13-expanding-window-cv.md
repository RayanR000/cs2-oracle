# Expanding-Window Cross-Validation — 2026-07-13

Replaced the single 21-day holdout validation in `train()` with expanding-window CV. Confidence thresholds are now calibrated on pooled out-of-fold (OOF) predictions across multiple time periods, giving more stable and representative calibration.

This was the last remaining pending item from the model audit (`model-audit.md:400`).

---

## Problem

The `train()` method used a **single 21-day holdout** at the end of the data for:
1. Optuna hyperparameter search
2. Confidence threshold calibration
3. Accuracy measurement

A single split is sensitive to the specific events in that window (a Major during validation inflates error; a quiet period deflates it). The 30d horizon already showed fold std of 12.0% and a range of 37.2%–86.8%, meaning the single-split accuracy metric was highly dependent on which 21 days were held out.

---

## Changes

### `backend/models/forecaster.py`

**New class constants:**
- `CV_STEP_DAYS = 120` — days between each fold's validation window
- `CV_MIN_TRAIN_DAYS = 200` — minimum unique dates before first validation fold

**New instance variable:**
- `self.cv_results: Dict[int, Dict]` — per-horizon fold-level metrics, saved to `meta.json`

**New methods:**

`_compute_cv_splits(sorted_dates)` — splits sorted dates into 5 expanding-window folds:
```
Fold 1: train [0..200d] → val [201..221d]
Fold 2: train [0..320d] → val [321..341d]
Fold 3: train [0..440d] → val [441..461d]
Fold 4: train [0..560d] → val [561..581d]
Fold 5: train [0..680d] → val [681..701d]
```

`_cv_evaluate_horizon(tdf, horizon, per_quantile_params)` — runs CV for a single horizon:
- Trains a single LightGBM model per quantile per fold (200 rounds, best HP from Optuna, no ensemble)
- Applies quantile crossing fix and computes per-row `range_pct`/`change_pct`/`hit` records
- Collects fold-level directional accuracy metrics
- Returns pooled OOF records + fold metrics list

**Refactored method:**

`_calibrate_confidence()` — now accepts optional `records_df` parameter:
- `records_df` provided (CV path): skips prediction computation, goes directly to threshold optimization
- `records_df` omitted (fallback): uses existing prediction-from-ensemble code path

**Modified methods:**

`train()` — per horizon:
1. Saves per-quantile best params: `per_quantile_params[q] = dict(base_params)`
2. After ensemble training: calls `_cv_evaluate_horizon()` to get pooled OOF records
3. Calibrates confidence on pooled OOF records (or falls back to single-split if CV produced no records)
4. Logs fold-level metrics and stores in `self.cv_results[horizon]`

`save_models()` — serializes `cv_results` to `meta.json` under a `"cv_results"` key.

---

## Data Flow

```
Training Data (730d)
    │
    ├── Single Split (last 21d) ──→ Optuna HP search (unchanged)
    ├── Full data ──→ Ensemble training (unchanged, 3 seeds × 1000 rounds)
    │
    └── Expanding-Window CV ──→ 5 folds, 120-day steps
              │
              ▼
         Pooled OOF predictions (range_pct, change_pct, hit)
              │
              ▼
         Confidence threshold calibration
              │
              ▼
         Fold-level accuracy metrics ──→ logged + saved to meta.json
```

---

## Runtime Impact

| Stage | Rounds | Time Estimate |
|-------|--------|---------------|
| Optuna search | 72,000 (unchanged) | ~10 min |
| Final ensemble | 36,000 (unchanged) | ~5 min |
| CV evaluation (new) | 12,000 | ~2-3 min |
| **Total** | **120,000** | **~17-18 min** |

CV adds ~15% overhead. Well within the 120-min GHA timeout.

---

## Verification

- All 41 existing unit tests pass unchanged
- `_compute_cv_splits` produces 5 folds with correct 21-day validation windows
- `_calibrate_confidence` works with both `records_df` and fallback (X_val/y_val/val_set) paths
- `save_models` serializes cv_results alongside existing meta
