# Retrain Optimization Analysis

Estimates and side-effect analysis for each proposed change to the training pipeline.
Measured on MacBook Pro (10 cores, no GPU). Source: `backend/models/forecaster.py`.

> **Update (2026-07-26):** `SKIP_HP_HORIZONS` narrowed from `[3, 14, 30]` to `[3]` — 14d/30d
> DART now run Optuna HP search (15 trials) instead of `DART_PARAMS` defaults, and `subsample`
> is tuned on the bagging branch. Measured retrain with `SKIP_REGIMES=1 FORCE_HP_SEARCH=1`:
> **16m16s** wall-clock (`user 3115s` / `real 976s` ≈ 3.2× parallelism). See
> [Measured Results: 14d/30d HP Search](#measured-results-14d30d-hp-search-2026-07-26) below.

---

## Current Retrain Times

| Scenario | Time Estimate | Basis |
|----------|--------------|-------|
| **Cold retrain** (full: HP search + CV + regimes + feature validation) | **14-16 min** | All phases enabled |
| **Warm retrain** (only ensemble training: 36 models) | **4-5 min** | No HP/CV/regimes/feat val |
| **Prediction only** (load models) | **1-2 min** | Read cache + predict |

### Phase breakdown (cold retrain)

| Phase | Time |
|-------|------|
| Build data (DuckDB query, 200K stratified subsample, ~70 features) | ~50s |
| 3d GBDT ensemble + CV + regimes + feat val | ~170s |
| 7d GBDT ensemble + Optuna (15 trials) + CV + regimes + feat val | ~210s |
| 14d DART ensemble + CV + regimes + feat val | ~220s |
| 30d DART ensemble + CV + regimes + feat val | ~220s |
| Save models | ~5s |

Training data: 4-year window has 9.1M rows, 41K items, 1,372 distinct dates.
CV produces ~10 folds (step=120, min_train=200, val_window=21).

---

## Change Analysis

### 1. `SKIP_REGIMES=1`

**Time saved: ~3.5 min** (54s per horizon × 4, subset-dependent)

**Side effects:**
- Regime models never trained or loaded
- Predict falls back cleanly at `forecaster.py:2738`: `"no regime models trained, using global"`
- The global fallback path is already exercised whenever regime data is insufficient (`MIN_REGIME_TRAIN=500`)
- No A/B test data exists comparing regime vs global — `ab_test_regime.py` was written but never run (zero `ab_test_regime` records in prediction_accuracy parquet)
- The regimes/global usage ratio log at `forecaster.py:2894` always shows 0% regime — cosmetic

**Verdict:** Safe. Zero accuracy impact in practice.

---

### 2. `SKIP_CV=1`

**Time saved: ~4 min** (63s per horizon × 4)

**Side effects:**
- Conformal calibration (`q_hat`) is NOT computed — `conformal_calibration[horizon]` never set (line 2391-2400)
- Predict uses `q_hat=0` (line 2817) → intervals are not widened by CQR
- Single holdout calibration (21-day window) used instead of pooled OOF from ~10 CV folds

**Asymmetric impact by horizon.** Actual coverage with CV enabled:

| Horizon | Boosting | High Coverage (CV on) | Risk with SKIP_CV |
|---------|----------|----------------------|-------------------|
| 3d | GBDT | 48% | Low — already broken |
| 7d | GBDT | 39% | Low — already broken |
| 14d | DART | 91% | **High** — would lose CQR widening |
| 30d | DART | 92% | **High** — would lose CQR widening |

The DART horizons (14d, 30d) have working conformal calibration (~91% coverage). SKIP_CV=1 would drop them to ~50% (raw, unwidened intervals). The GBDT horizons are already at 39-48% — interval coverage is broken regardless.

**Verdict:** High risk for DART horizons. Mitigation: only set `SKIP_CV=1` for GBDT horizons (3d, 7d). Or skip entirely — the 4 minutes is already partially offset by other changes.

---

### 3. Cap GBDT boost rounds at 500 (from 1000)

**Time saved: ~20s** (only affects q10 models, ~5 models hit >500)

**Side effects:** Measured from actual saved models:

| Quantile | Median rounds | Max rounds | Hit 1000 cap? |
|----------|--------------|------------|---------------|
| q10 (GBDT) | **774** | 1000 | Yes |
| q50 (GBDT) | **359** | 420 | No |
| q90 (GBDT) | **1** | 3 | No |

- q10 models genuinely need >500 rounds — they're still improving at 774 median. Capping at 500 would truncate them and degrade the lower prediction interval.
- Early stopping already terminates q50/q90 far before 500. The ~5 models hitting 774-1000 are 3d/7d q10. Each saves ~274 rounds × 0.016s = ~4s. Total: ~20s.
- DART is already fixed at 500 rounds (no early stopping) — unaffected.

**Verdict:** Not worth the accuracy cost. Saves ~20s, degrades q10 models.

---

### 4. Drop social features

**Time saved: ~2s** (feature engineering only)

**Side effects:**
- 5 features removed: `social_mentions_1d`, `social_mentions_7d`, `social_mention_velocity`, `social_sentiment_7d`, `social_score_7d`
- No downstream code references them by name (feature grouping is dynamic)
- AGENTS.md explicitly documents: "Social sentiment features are non-functional. VADER scores CS2 jargon as neutral — features don't rank in top 20 by gain."
- The `/social-sentiment` API endpoint reads from its own parquet table — unaffected

**Verdict:** Safe but trivial gain. Only worth doing if touching this area anyway.

---

### 5. Reduce 7d Optuna trials from 15 to 10

**Time saved: ~10s** (5 fewer trials × ~2s each)

**Side effects:**
- `optuna_horizons_search.py` ran 50 trials per horizon — results showed consistent parameter ranges
- The 15-trial default was already a deliberate reduction from the 50-trial gold standard
- Hyperband pruner (min_resource=5, max_resource=200, reduction_factor=3) creates 4 brackets. At 10 trials, some brackets won't be fully populated, but pruner adapts.
- Only affects 7d horizon (3d/14d/30d are in `SKIP_HP_HORIZONS`)

**Verdict:** Marginal. ~10s saved, small chance of suboptimal hyperparams. The 15-trial default was already a compromise.

---

### 6. Halve `max_feature_rows` from 200K to 100K

**Time saved: ~2-4 min** (each model trains ~2× faster, 36 models)

**Side effects:**
- Items kept: ~900 (at 200K) → ~450 (at 100K), stratified by rarity
- Calendar window preserved (1,372 dates) — CV still produces ~10 folds
- Full item histories kept (not `tail()` truncation) — lag/rolling features stay valid
- Correlation pruning at `PRUNE_CORRELATION_THRESHOLD=0.95` is stable at 100K rows (variance of ρ estimates ≈ (1-ρ²)²/(n-1), negligible at n=100K)
- 2026 data exclusion already drops some items (~352 noted in code comment) — combined with 100K may further reduce coverage of rare items

**Verdict:** Acceptable risk for the time savings (~2-4 min). If you're concerned about rare item generalization, keep at 200K and just accept the slower training.

---

## Bonus Findings

### q90 GBDT models are broken

Every GBDT q90 (alpha=0.9) model terminates at **1-3 boost rounds**. This is the root cause of the 39-48% interval coverage on GBDT horizons.

**Root cause:** `data_sample_strategy="goss"` with `top_rate=0.2, other_rate=0.1` is incompatible with quantile regression at extreme quantiles (alpha=0.9). The gradient distribution causes GOSS to select almost no high-gradient samples, so early stopping fires immediately.

**Fix:** Disable GOSS for q != 0.5, or switch to `"bagging"` for q10 and q90.

This would likely produce a larger accuracy improvement than any of the optimization changes above.

---

## Measured Results: 14d/30d HP Search (2026-07-26)

Ran `SKIP_REGIMES=1 FORCE_HP_SEARCH=1 python scripts/forecast_prices.py --train-only` with
`SKIP_HP_HORIZONS=[3]` (previously `[3, 14, 30]`), on the same 10-core Mac, full 700K-row /
1460-day window (production config, not the 200K subsample used for the estimates above).

**Time: 16m16s real** (`user 3115s`, `sys 553s` — ~3.2× OpenMP parallelism). Faster than the
14-16 min cold-retrain estimate despite adding two new 15-trial Optuna searches, because
`SKIP_REGIMES=1` more than offset the added HP search cost.

**Accuracy — before (old cached HP) vs. after (new Optuna-tuned HP), from `meta.json`
`cv_results` / `conformal_calibration`:**

| Horizon | Old dir-acc | New dir-acc | Edge vs. baseline (old→new) | Conformal q̂ (old→new) |
|---|---|---|---|---|
| 3d | n/a¹ | 64.0% ±5.9 (9 folds) | — → +9.1pp | — → 1.38 |
| 7d | n/a¹ | 61.8% ±7.8 (9 folds) | — → +8.1pp | — → 1.36 |
| **14d** *(tuned)* | 40.6% ±7.3 (4 folds) | 58.4% ±8.1 (9 folds) | +8.4 → +7.9pp | 8.53 → **6.86** |
| **30d** *(tuned)* | 42.2% ±19.0 (4 folds) | 58.9% ±8.4 (8 folds) | +10.5 → +3.8pp | 13.34 → **5.69** |

¹ Old model had `SKIP_CV=1` on GBDT horizons, so it recorded zero 3d/7d folds — no comparable baseline.

**Read:**
- **Calibration improved cleanly.** Conformal q̂ (interval widening needed to hit 90% coverage)
  dropped on both tuned horizons — 30d roughly halved (13.34→5.69). This is fold-count-independent
  and is the clearest evidence the new HP search helped.
- **Directional accuracy is not a clean comparison.** Old run used 4 CV folds, new run uses 8-9 —
  different validation periods, not apples-to-apples. The raw +17pp deltas are not attributable to
  the HP change alone.
- **Edge vs. baseline (fold-independent) is mixed.** 14d held roughly steady (+8.4→+7.9pp) but 30d's
  edge shrank (+10.5→+3.8pp) even as its calibration improved.
- **Verdict:** safe to keep — better-calibrated intervals, no directional regression vs. baselines —
  but not yet a rigorously attributed accuracy gain. A same-fold-count A/B (old vs. new
  `SKIP_HP_HORIZONS`, same seed/window) would be needed to isolate the effect, following the pattern
  of the shelved quality-spread A/B (`docs/changelog`, commits `b7188e3`/`12d6c5e`).

---

## Recommended Plan

| Priority | Change | Time Saved | Accuracy Risk | Notes |
|----------|--------|-----------|---------------|-------|
| 1 | Fix q90 GOSS bug | (slight slowdown) | **Accuracy gain** | Fixes broken upper intervals |
| 2 | `SKIP_REGIMES=1` | ~3.5 min | None | Safe, clean fallback |
| 3 | `SKIP_CV=1` for GBDT only | ~2 min | Low | Already broken on 3d/7d |
| 4 | `max_feature_rows` = 100K | ~2-4 min | Low-Med | Keep full calendar window |
| 5 | Drop social features | ~2s | None | Trivial, safe |
| 6 | 7d Optuna → 10 trials | ~10s | Low | Marginal gain |
| 7 | Cap GBDT at 500 rounds | ~20s | Medium | Don't do this |

**Target cold retrain with changes 1-4:** ~8-10 min (down from 14-16). Changes 2-4 are still
open; only the `SKIP_HP_HORIZONS` narrowing (7d→7d/14d/30d) has been applied so far — see
[Measured Results](#measured-results-14d30d-hp-search-2026-07-26).
**Target warm retrain:** ~3-4 min (down from 4-5).

### Env var command (without code changes):

```bash
# Cold retrain with safe skips:
SKIP_REGIMES=1 SKIP_CV=1 python scripts/forecast_prices.py --train-only

# Force full HP search (7d/14d/30d — 3d stays frozen via SKIP_HP_HORIZONS) on next
# Sunday or after data events:
SKIP_REGIMES=1 SKIP_CV=1 FORCE_HP_SEARCH=1 python scripts/forecast_prices.py --train-only
```

Note: `SKIP_CV=1` is unsafe to combine with `FORCE_HP_SEARCH=1` on 14d/30d — CV drives their
conformal calibration (see [item 2](#2-skip_cv1) above), which is the main thing the 2026-07-26
run improved. Use `FORCE_HP_SEARCH=1` without `SKIP_CV=1` when re-tuning those horizons.
