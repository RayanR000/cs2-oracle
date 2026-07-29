# Model Optimization Options

> Covers all levers to make the model smaller, faster, and cheaper to train/infer
> while retaining ≥90% of current accuracy (DA within ~5pp of baseline).
> Source of truth: `backend/models/forecaster.py:ItemForecaster` class constants.

---

## Current Baseline (2026-07-29)

| Metric | Value |
|--------|-------|
| **Models** | 36 global (4H × 3Q × 3E) + ≤108 regime (3R × 4H × 3Q × 3E) = **36–144 LightGBM models**. The 2026-07-29 retrain ran `SKIP_REGIMES=1`, so the current on-disk set is **40 files** (36 global + 4 directional classifiers, no regime models). `predict()` falls back to global cleanly and logs `no regime models trained, using global` |
| **Ensemble size** | `N_ENSEMBLES = 3` (seeds 42, 73, 91; feature fractions 0.6, 0.7, 0.8) |
| **Horizons** | 3d (GBDT), 7d (GBDT), 14d (DART), 30d (DART) |
| **Quantiles** | p10 (0.1), p50 (0.5), p90 (0.9) — all bagging as of 2026-07-29. **Correction:** this row previously read "p90 GBDT is broken (1-3 rounds, GOSS incompatibility)" — that was backwards. p90 was the healthiest GBDT quantile; **q50** was the collapsed one (7d q50 saved 1–2 trees), because GOSS was applied to q50 only and is degenerate under the quantile objective's constant ±alpha gradients. Fixed — see `docs/changelog/2026-07-29-q50-row-sampling.md` |
| **Boost rounds** | GBDT: 1000 (early stop 50), DART: 500 (no early stop) |
| **Features** | ~70–120 after correlation pruning (threshold 0.95), 8 groups |
| **Rows** | `max_feature_rows = 100K` (stratified subsample) |
| **HP search** | 3d skipped (frozen, 50-trial winner), 7d=10 trials, 14d=15 trials, 30d=15 trials (as of 2026-07-26 — see below) |
| **Warm retrain** | ~4–5 min (HP cached) |
| **Cold retrain** | **12m30s** measured 2026-07-29 (`SKIP_REGIMES=1 FORCE_HP_SEARCH=1`, CV on, 3d HP frozen). ~14–16 min with regimes |
| **Inference** | ~1–2 min with a warm 3-day feature cache; **~5 min on a cold cache** (rebuilds 6.1M voted rows from Parquet) |
| **Production DA** | ⚠️ **STALE — measured pre-q50-fix.** 3d=61.5%, 7d=52.8%, 14d=55.7%, 30d=54.2% (+35–41pp vs baseline). These predate the 2026-07-29 q50 sampling fix and cannot be refreshed on demand: `backtest_accuracy.py` scores *stored* forecasts against matured actuals, so the current model will not appear in it until the forecasts written 2026-07-29 mature (3–30 days). For a fresh-model number, run `scripts/walkforward_backtest.py` (~60–90 min, all horizons) |
| **Classifier CV DA** | 3d=68.2%, 7d=68.3%, 14d=68.8%, 30d=70.9% (9 folds; 8 for 30d) — measured during the 2026-07-29 retrain. This is the **served** direction signal (classifier, not quantile-sign, since 2026-07-24). A training-time diagnostic, **not** comparable to the Production DA row above |
| **Quantile-sign CV DA** | 3d=61.0%, 7d=61.0%, 14d=60.8%, 30d=64.0% (same run; sd 5.2–8.9%) |

---

## Optimization Levers

### Tier 1 — Zero quality risk

| # | Lever | Change | Speed Gain | Model Reduction | Quality Impact | Implementation |
|---|-------|--------|-----------|----------------|----------------|----------------|
| **1** | Drop social features | Remove 5 VADER features from `engineer_features()` | ~2s feature engineering | — | 0pp (not in top 20) | Delete `social_` from `_add_social_features()` or skip call |
| **2** | Skip regime models permanently | Set `SKIP_REGIMES=1` as default | ~3.5 min train / ~0s infer | −108 models (max) | 0pp (0% regime usage logged) | Change `forecaster.py` env check to default-true or hardcode regime skip |
| **3** | Reduce ensemble 3→2 | `N_ENSEMBLES = 2`, seeds=[42,73], fractions=[0.6,0.7] | ~33% ensemble training | −12 models (36→24) | ~0.3–0.5pp (ensemble variance shrinks) | Edit class constants |
| **4** | Reduce `max_bin` 63→31 | `MAX_BIN = 31` | ~10–15% per fit | — | ~0.3–0.5pp (coarser splits) | Edit class constant. Was 255, already halved once |
| **5** | Drop `_validate_feature_groups` permutation test | Skip permutation pruning entirely | ~2–5% per horizon | — | 0pp (diagnostic only) | Return early in `_train_horizon_ensemble` |

**Combined Tier 1:** Models 36→24, warm retrain ~2–3 min, cold ~8–10 min, inference ~45–60s. Quality: −0.6–1.0pp DA (well within 90% retention).

---

### Tier 2 — Low quality risk

| # | Lever | Change | Speed Gain | Model Reduction | Quality Impact | Implementation |
|---|-------|--------|-----------|----------------|----------------|----------------|
| **6** | Reduce `max_feature_rows` 100K→50K | Half training rows | ~2–3 min per train | — | ~0.3–0.7pp (fewer items sampled) | Change default parameter |
| **7** | Reduce `num_leaves` 47→31 | Simpler trees | ~15–20% per fit | — | ~0.3–0.5pp | Edit class constant (was 31 in old docs) |
| **8** | Skip 7d HP search | Add 7 to `SKIP_HP_HORIZONS` | ~2 min (15 saved trials) | — | ~0.1–0.3pp (uses fallback params) | `SKIP_HP_HORIZONS = [3, 7, 14, 30]` |
| **9** | Reduce Optuna trials 15→10 for 14d/30d | `N_TRIALS_MAP[14]=10, [30]=10` | ~1 min | — | ~0.1pp | Edit N_TRIALS_MAP |
| **10** | Parallel ensemble training | ThreadPoolExecutor 2-wide | ~5 min (on cold retrain) | — | 0pp (identical training) | Implement in `_train_horizon_ensemble` (noted as safe after Dataset construction) |
| **11** | Drop horizon-excluded features at source | Skip computing cross-sectional/event features for 14d/30d | ~5–10s feature engineering | — | 0pp (already excluded from model) | Conditional in `engineer_features()` |

---

### Tier 3 — Measurable quality risk

| # | Lever | Change | Speed Gain | Model Reduction | Quality Impact |
|---|-------|--------|-----------|----------------|----------------|
| **12** | Drop 14d horizon | `HORIZONS = [3, 7, 30]` | ~25% training + inference | −9 models (36→27) | Loses 55.7% DA horizon (+38pp baseline). Consider keeping DART-only if used for trend confirmation |
| **13** | Cap GBDT rounds at 500 | `num_boost_round = 500` for GBDT | ~20s (only q10 affected) | — | Medium — q10 still improving at 774 median. Truncates lower interval |
| **14** | Single model per (horizon, q) | `N_ENSEMBLES = 1` | ~66% ensemble training | −24 models (36→12) | ~0.5–1.0pp (no averaging stabilization) |
| **15** | Aggressive correlation pruning | `PRUNE_CORRELATION_THRESHOLD = 0.85` | ~5–10% per fit | Fewer features | ~0.3–0.5pp (loses some signal-bearing correlated features) |
| **16** | Increase regularization | `lambda_l2 = 2.0`, `min_data_in_leaf = 30` | ~5% per fit | Smaller trees | ~0.3–0.5pp |

---

### Tier 4 — Major architectural changes

| # | Lever | Change | Speed Gain | Model Reduction | Quality Impact |
|---|-------|--------|-----------|----------------|----------------|
| **17** | Drop p10/p90 quantiles | `QUANTILES = [0.5]` only | ~66% training + inference | −24 models (36→12) | Loses prediction intervals. Frontend uses `forecast_low/forecast_high` — needs UI change. **Correction (2026-07-29):** this cell previously read "p90 GBDT already broken (1-3 rounds)" — same error as the Quantiles row above, and false. Post-retrain p90 trains healthily (3d 330/305/281, 7d 270/188/85) and so does p10 (3d 226/176/174, 7d 286/237/206). Interval coverage on GBDT is genuinely poor (39–48% raw vs 80% nominal), but that is a calibration problem, not a collapsed model — and conformal q̂ already corrects it at serve time |
| **18** | LightGBM → ONNX conversion | Convert to ONNX for inference | 0 train impact, ~2–5× faster inference | Smaller on-disk | ~0pp (if conversion is precise) |
| **19** | Replace with CatBoost | CatBoost (tested Jul 2026, degraded 18-20pp) | — | — | **Already tested and rejected** |
| **20** | Replace with neural forecast | N-BEATS / PatchTST / TFT | Slower training | — | Unknown, requires GPU |

---

## Recommended Implementation Order

### Quick win (1 session, ~20 min of code changes):

1. **Drop social features** — delete `social_` column generation
2. **Skip regime models permanently** — flip default in code
3. **Reduce ensemble 3→2** — edit `N_ENSEMBLES`, `ENSEMBLE_SEEDS`, `ENSEMBLE_FEATURE_FRACTIONS`
4. **Reduce `max_bin` 63→31** — edit `MAX_BIN`
5. **Add 7 to `SKIP_HP_HORIZONS`** — edit class constant
6. **Skip feature validation on all retrains** — remove permutation test

**Result:** 36→24 global models, no regimes saved/loaded, ~2 min warm retrain, ~30s inference.

### If intervals are expendable (2 sessions):

7. **Drop p10/p90** — `QUANTILES = [0.5]`. 24→12 models. Update frontend to remove interval display. Current intervals are already partially broken.

### If you want maximum speed (separate session):

8. **Drop 14d horizon** — 12→9 models
9. **Reduce `max_feature_rows` 100K→50K** — ~2× faster per fit
10. **Parallel ensemble training (2-wide)** — ~5 min saved on cold retrain

---

## Verification Protocol

After applying any change, verify:

```bash
cd backend
pytest tests/test_forecaster.py -x -q                    # unit tests pass (114 as of 2026-07-29)
SKIP_REGIMES=1 FORCE_HP_SEARCH=1 python scripts/forecast_prices.py --train-only   # retrain succeeds
python scripts/walkforward_backtest.py                    # fresh-model DA (~60-90 min)
```

**Do not use `SKIP_CV=1` when the quantile models change.** Conformal q̂ is
derived from CV out-of-fold predictions (the CQR block that populates
`ItemForecaster.conformal_calibration`), so skipping CV leaves a stale q̂
calibrated against differently-shaped quantiles.

**`--predict-only` can trigger an unwanted retrain.** It drift-checks every
horizon and auto-retrains via `forecaster.train(max_rows=700_000)` — with no
`SKIP_REGIMES` — if any horizon is below the 60% threshold. Drift is scored
from *stored* forecasts, so a recent bad patch (e.g. the 2026-07-29 synthetic
overwrite, which left 3d at 37.2% and 7d at 39.3%) will retrain a model that
is minutes old. Check drift read-only via `forecaster.check_concept_drift`
before running predict on a fresh model.

**Which accuracy script to use:**

| Script | Measures | Reflects a fresh model? |
|---|---|---|
| `walkforward_backtest.py` | Retrospective walk-forward folds using current tuned params | **Yes** — the fresh-model gate |
| `backtest_accuracy.py` | Stored forecasts vs matured actuals | **No** — needs 3–30 days for forecasts to mature. This is what CI runs |

| Horizon | Baseline DA (pre-q50-fix) | 90% Retention Floor |
|---------|--------------------------|---------------------|
| 3d      | 61.5%                    | ≥55.4%              |
| 7d      | 52.8%                    | ≥47.5%              |
| 14d     | 55.7%                    | ≥50.1%              |
| 30d     | 54.2%                    | ≥48.8%              |

---

## Previous Optimizations Already Applied

These are NOT new levers — they're already in production:

| Optimization | Value | Previously |
|-------------|-------|------------|
| `N_ENSEMBLES` | 3 | Was 6 (was 9) |
| `MAX_BIN` | 63 | Was 255 |
| `max_feature_rows` | 100K | Was 700K (was 400K) |
| `SKIP_HP_HORIZONS` | [3] | Was [3, 14, 30] (2026-07-26: re-enabled 15-trial Optuna search for 14d/30d DART; measured retrain 16m16s with `SKIP_REGIMES=1 FORCE_HP_SEARCH=1` — improved conformal calibration, see `docs/retrain-optimization-analysis.md`) |
| `N_TRIALS_MAP[3]` | 50→20 (then 50) | Iterated |
| Feature cache | 3-day TTL | Was none |
| Row sampling | **bagging for all quantiles** (2026-07-29) | Was q50=GOSS, q10/q90=bagging — GOSS reverted after A/B showed +5.21% (3d) / +1.76% (7d) pinball and +1.13pp / +0.71pp DA for bagging |
| 2026 data exclusion | Active | Was all data |
| Dead item filter | <$0.05, <5% range | Was none |
| Target winsorization | ±500% clip | Was none |
