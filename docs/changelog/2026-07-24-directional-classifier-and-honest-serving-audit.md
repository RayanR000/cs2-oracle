# Directional classifier, price-only simplification, and honest serving audit

**Date:** 2026-07-24

**Branch:** `simplify-price-only-model` (not yet merged)

**Files changed:**
- `backend/models/forecaster.py` — honest CV harness (purge gap + naive baselines), price-only feature allowlist, longer-trend features, 3-class directional classifier (train/CV/save/load/serve), stale-model predict guard, date-based (gap-robust) lag/return features. Momentum fallback added then retired.
- `backend/scripts/forecast_prices.py` — construct forecaster with `prune_failed_groups=False`.
- `backend/tests/test_forecaster.py` — +~25 tests (baselines, purge gap, allowlist, momentum/direction recenter, classifier helpers, ensemble-safety guard, gap-robust lags, trend features). 105 pass.
- `docs/changelog/2026-07-24-...md` — this entry.
- Moved 108 stale 49-feature regime model files to `backend/models/saved_models_stale_regimes_20260724/`.

---

## What / why

A model critique turned into an end-to-end overhaul. Each step was gated by an experiment, and two proposed improvements were correctly *rejected* by the data. The headline: the model now optimizes direction directly and beats naive baselines at every horizon in cross-validation — but a serving audit found the live path is compromised by an upstream **data gap**, so the CV win is not yet a validated *live* win.

## 1. The critique: raw accuracy was meaningless

The saved `meta.json` reported 14d directional accuracy of 32.7% — but this was a debug/fast-run artifact (~5 items, `n_val≈105`). Two real methodological problems were found:

- **Temporal leakage in CV.** Expanding-window folds put `val_start` one day after `train_end`, but an H-day training row's target lands H days later — *inside* the validation window. Fixed with a **purge/embargo gap of `horizon` days** in `_compute_cv_splits(purge_days=…)`.
- **No baseline.** Directional accuracy is meaningless without a benchmark. Added **persistence** (predict-flat / random walk) and **momentum** (sign of trailing `return_Nd`) baselines to every CV fold, plus `edge_vs_best_baseline` in `cv_results`, and a warning when the served model fails to beat them.

CV was also widened: `VALIDATION_WINDOW_DAYS` 21→30, `CV_STEP_DAYS` 200→150 → ~7 folds of 30 days each on the ~1,256-date archive.

## 2. Feature ablation → price-only

With honest CV, a full-vs-price-only ablation (126 → 41 features) showed the **85 non-price features (events, social, cross-sectional, supply, item-identity, temporal) add ~0 directional accuracy and hurt at 3d/30d** (full−price = −0.6 / +0.1 / +1.6 / −1.3pp, all within fold noise). Added `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]`, applied in `build_training_data`.

## 3. Two rejected experiments (the data said no)

- **Model+momentum blend:** a weight sweep found the optimum is always a corner (pure model or pure momentum) — blending never beats the better single signal. Rejected.
- **Longer-lookback trend features** (`return_90/120/180d`, distance-from-200d-MA, trend consistency): moved directional accuracy by −0.6/+0.1/+1.6/−0.6pp, all noise. The features were *kept* only because they later help the classifier; on their own they do nothing.

Both negatives pointed at the same root cause: **the problem was the objective, not the features.** The quantile model optimizes return *magnitude*, then reads direction off the p50 sign — it shrinks toward conservative near-zero predictions and loses the directional call.

## 4. The fix that worked: a directional classifier

Added a per-horizon **3-class (down/flat/up) LightGBM classifier** trained on multiclass log-loss — optimizing the served metric directly — with training **up-weighted toward movers** (`DIRECTION_MOVER_WEIGHT = 3.0`) so capacity goes to the hard up/down calls rather than the easily-predicted flat mass. The quantile models still supply the price interval; `predict()` takes direction + confidence from the classifier and recenters the median on its call (preserving interval width). Momentum fallback retired (`MOMENTUM_FALLBACK_HORIZONS = []`).

**7-fold CV (2022–2025), served = classifier:**

| Horizon | Classifier | Quantile-sign (old) | Momentum | Edge vs best baseline |
|---|---|---|---|---|
| 3d | 69.8% | 64.9% | 59.9% | +9.9pp |
| 7d | 69.7% | 63.3% | 61.1% | +8.6pp |
| 14d | 68.1% | 62.4% | 61.7% | +6.4pp |
| 30d | 69.2% | 60.2% | 67.2% | +2.0pp |

First approach to beat momentum at **every** horizon, including 30d.

**Honest caveat:** ~40% of item-returns are "flat" (static items, predictable from low volatility), so much of the ~69% is correctly calling non-movers. Sign skill on items that *actually move* (|return| > 0.5%) is only ~55–57% — real, above chance, but modest. Mover-weighting shifts capacity toward those calls without hurting overall accuracy.

## 5. Serving audit — a crash bug and the real blocker

A `predict()` smoke test (which the unit tests don't cover with real models) caught two things:

- **Stale-model crash (fixed).** The model dir held 108 regime-model files from a superseded 49-feature run; `predict()` fed 46-feature data into them → `LightGBMError`. Added `_predict_ensemble_safe()` (skips any model whose feature count ≠ the current matrix, falls back regime→global; same guard on residual models) and moved the stale files out. Prediction now degrades gracefully across any feature-schema change.
- **Skewed serving distribution → the real problem.** Live forecasts were ~67–71% "down" with near-zero "up" (7d: 0%), and ~19% showed a direction disagreeing with a near-flat price. Investigation:
  - **Not corruption** (option 3, ruled out): 2026 daily moves are ~2× hotter but only 26% of >100% jumps revert next day (vs 81% in 2022–2025 = the corruption signature); corrupt items are 2.8% of rows; full cleaning leaves 2026 at +70% mean / 60% up.
  - **A data gap** (the real blocker): **May–June 2026 are entirely missing and July is partial.** Trailing-return features used a **row-based** `shift(lag)` that spans the hole, turning `return_14d/30d` at the serving edge into ~3-month returns (the +70% means) — wildly out-of-distribution, collapsing the classifier toward flat/down.

## 6. Gap-robust features (option B)

Converted all lag/return features in `_compute_price_features` to **date-based** lookups (a date-keyed self-merge, exactly what `prepare_targets` already did for targets): the price is looked up `N` *calendar* days earlier, and a missing date yields NaN → imputed to neutral instead of a fabricated jump. Also re-bound the stale post-merge `groupby` handle and de-duped the lookup keys.

This is correct and tested, but re-checking serving showed the distribution **barely moved** — because (a) the *rolling* technicals (RSI/MACD/Bollinger/MA/z-score/support-resistance) are still row-based and span the gap, and (b) more fundamentally, the model trains on 2022–2025 (2026 excluded) and serves a hotter 2026 regime with a hole in it. No feature-plumbing overcomes missing data + a train/serve regime mismatch.

## Status & open blockers

**Validated and keepable:** honest CV harness (purge + baselines), price-only simplification, directional classifier, stale-model crash guard, date-based gap-robust features. 105 tests pass; cold retrain ~10.5 min (`FORCE_HP_SEARCH=1 SKIP_REGIMES=1`).

**Not production-ready — serving is gated on:**
- **(A) Data gap — operational, top priority.** Find why May–June 2026 is missing / July partial (ingestion or archive-export). Nothing downstream is reliable without fresh continuous data.
- **(C) Regime mismatch.** After (A): stop excluding recent data, retrain on the current regime, and validate walk-forward *through* the serving window. The ~69% above is a 2022–2025 CV number and overstates live performance.

**Do not read the ~69% as a live figure.** It is CV on held-out 2022–2025 data. Live directional accuracy is unmeasured pending (A)+(C).
