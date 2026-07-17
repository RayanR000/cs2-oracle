# Prediction Model — Priority Fixes

**Created:** 2026-07-17
**Status:** Planning (not yet implemented)
**Context:** Warm retrain is ~5.7 min against a 20–30 min budget. The model is **not time-constrained — it is correctness-constrained.** Spend the headroom on robustness, not more boosters.

## Current state (trustworthy numbers only)

Only the post-column-order-bug numbers from `docs/changelog/2026-07-17-retrain-and-backtest-baseline.md` are valid. Everything before 2026-07-17 was trained on `volume` mislabeled as `price`.

| Metric | Value |
|---|---|
| Training (warm, HP cached) | ~5.7 min |
| Training (cold, full Optuna) | ~30–45 min |
| 6-fold CV dir. accuracy | ~68% all horizons |
| Real backtest 3d / 7d / 14d / 30d | **60.6 / 62.1 / 53.4 / 42.5%** |
| MAE | $0.74–0.77 |
| Interval coverage | 43–60% |

Three problems drive this plan:
1. **7d regression is live and unfixed** — 62.1% → 53.0% after the dead-item-filter retrain.
2. **14d/30d models pruned to just 4 features** — same root cause as the 7d collapse.
3. **CV (68%) vs backtest (42–62%) gap** — CV is over-optimistic; last run had `fold_count=0` (silent fallback to single split).

Root cause common to #1 and #2: **the model makes pruning/calibration decisions off validation sets too small to be trustworthy.**

---

## Tier 1 — Fix what is actively broken (biggest wins, ~zero training-time cost)

### 1. Fix the 2026 distribution-shift guard
The guard in `build_training_data()` is **not actually excluding 2026 rows** — 35,441 remain. This pushes the 7d validation window into sparse mid-2026 data (a single calendar day, ~352 items), where the permutation test is pure noise.

- **File:** `backend/models/forecaster.py` (`build_training_data`, ~lines 1359–1367)
- **Investigation:** `docs/changelog/2026-07-17-7d-regression-investigation.md`
- **Three unverified hypotheses to check** (nail down which is real before fixing):
  - dtype / `.dt.year` issue on the date column
  - guard runs *after* the stratified subsample instead of before
  - silent no-op because the column isn't actually datetime
- **Expected impact:** recovers 7d ~53% → ~62%. Zero training-time cost.

### 2. Raise the validation-set floor
The current `val_set < 100 → fallback` threshold is far too low. Permutation tests on ~350 rows are noise.

- **File:** `backend/models/forecaster.py` (temporal split, ~lines 1468–1488)
- **Change:** require a meaningful minimum val size (e.g. ≥2–3K rows **or** a minimum number of distinct calendar days) before trusting permutation pruning or single-split calibration.
- **Expected impact:** prevents future thin-window collapses. Zero training-time cost.

### 3. Gate permutation-pruning conservatively
A model collapsing from 39 → 4 features should be a red flag, not accepted silently.

- **File:** `backend/models/forecaster.py` (`_validate_feature_groups`, ~lines 997, and the retrain-after-prune block ~1632–1680)
- **Change:** require the accuracy-drop signal to clear a confidence bound; skip pruning entirely when the val window is thin (ties into #2).
- **Expected impact:** restores 14d/30d to full feature sets. Zero training-time cost.

---

## Tier 2 — Close the CV ↔ backtest gap (spend the time budget here)

### 4. Make CV the metric we trust
`meta.json` showed `fold_count=0` last run — CV silently produced no folds and calibration fell back to a single split.

- **File:** `backend/models/forecaster.py` (`_compute_cv_splits` ~1070, `_cv_evaluate_horizon` ~2017)
- **Change:** run more/larger expanding-window folds; make a **zero-fold CV a hard failure**, not a silent fallback.
- **Cost:** more folds add time but stay well within the 20–30 min budget.

### 5. More Optuna trials on cold retrains
- **File:** `backend/models/forecaster.py` (`_optuna_search_params` ~1089)
- **Change:** 20 → 50 trials. HP results are cached, so this only costs time on the Monday cold run.
- **Expected impact:** +0.5–1pp (per `docs/2026-07-14-remaining-accuracy-improvements.md`).

---

## Tier 3 — Genuine accuracy levers (after Tier 1/2 land)

### 6. Regime-switching models
- Docs' top remaining item: +1–2pp avg, +2–4pp in volatile periods.
- **Cost:** ~+200% training time — still within 30 min.
- **Ref:** `docs/research/accuracy-opportunities.md` (#7)

### Do NOT pursue (docs already ruled these out)
- No neural models (N-BEATS/PatchTST) unless plateaued.
- Never revisit CatBoost (−18 to −20pp).
- Never add trade/sales volume (|r| < 0.002).
- Remember the calibration reality-check: proxied signals deliver ~10–20% of estimated gains; don't over-invest.

---

## Suggested order

1. Fixes **1 → 2 → 3** together (they share the thin-validation root cause), then retrain and re-run the Parquet backtest to confirm 7d recovers and 14d/30d regain features.
2. Then **4 → 5** to align CV with reality.
3. Only then evaluate **6**.

**Bottom line:** Don't add model capacity. Fix the validation/pruning pipeline first — it likely recovers several points across horizons at essentially no training-time cost.
