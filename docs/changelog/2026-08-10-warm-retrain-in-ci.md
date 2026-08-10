# CI training runs reuse hyperparameters, and the warm path stops dropping regime models

**Date:** 2026-08-10
**Implements:** levers 1 and 4 of `docs/research/2026-08-10-training-cost-levers.md`.
**MEASURED in CI: run `31405829674`** (`train-only`, `020a662`, x86) — **1117.9s training /
19m52s job**, against `31356483719`'s 2306.0s / 39m32s. First run under the 30-minute cap.

| Phase | `31356483719` | `31405829674` | Δ |
|---|---|---|---|
| Optuna | 692.9s | **0.0s** | **−692.9s** |
| Conformal CV | 514.1s | 392.8s | −121.3s |
| q50 ensemble | 309.3s | 232.2s | −77.1s |
| Direction classifier | 268.2s | 214.1s | −54.1s |
| Remainder | ~521s | ~279s | ~−242s |
| **TOTAL** | **2306.0s** | **1117.9s** | **−1188.1s (−51.5%)** |

**Do not quote −51.5% as this change's effect.** Only **−692.9s is attributable and
mechanism-verified**: all four horizons log `optuna: 0.0s (skipped - cached HP)`. Of the rest,
the remainder drop is largely the `voted-v6-` prefix restore (this run votes warm; run
`31356483719` built that frame cold — the confound was flagged before dispatch), and the three
fit phases came in 18–25% cheaper at *identical* cached HP and round counts, which sits inside
the ±25% per-phase run-to-run swing this project already documents. Read the total as "a warm
retrain on a warm cache costs ~1120s", not as a controlled delta.

**The regime correction was load-bearing, now with evidence.** The log shows `Loaded 4 global +
4 regime model groups ({'range'})` and then `regime models for {3,7,14,30}d done`. Under the old
`_warm_retrain` coupling those four restored groups would have been *deleted* and the artifact
shipped without them — and since `predict` prefers the regime model whenever the detected regime
matches, the served mid would have moved. Only the `range` regime qualifies (bear and bull hold
0 rows), and it covers 818K–893K of ~985K rows, so it is close to the whole cohort rather than a
niche branch.

## What changed

**`price-forecast.yml`: the model cache is restored on every mode**, not just `predict-only`.
`tuned_params` is populated only by the meta.json load path (`forecaster.py:6632-6645`), so a
training run starting with an empty `saved_models/` can never satisfy `reuse_hp` at `:4059`
and Optuna re-searches from scratch. That cost **692.9s** in run `31356483719` and 63.3s in
`31337078991`.

**`FORCE_RETRAIN=1` on `full` runs, which the restore makes mandatory.** `full` reaches the
age gate at `forecast_prices.py:390-402`, which retrains on model age alone (14 days). Monday's
weekly retrain runs 7 days after the last one, so a restored artifact reads as *fresh* and
`full` would have silently degraded into a predict-only run. This is the same trap that forced
the 2026-08-10 re-vote retrain to be dispatched as `train-only`
(`docs/changelog/2026-08-10-post-revote-retrain.md`). Both halves are pinned by one test, so
neither can be reverted alone.

**`ENGINEERED_CACHE=0` in CI.** `_save_engineered_cache` now honours the flag, mirroring
`VOTED_CACHE=0`. The workflow excludes this ~2GB frame from both the cache save and the
restore, so every run wrote one that nothing would ever read — `predict` only consults it
*before* feature engineering, never after. Local runs keep it; there it is a real speedup.

**`_warm_retrain` no longer skips regime-model training.** This is the correction, not a
saving — see below.

## The research doc was wrong twice, and both errors mattered

**1. "No effect on the served forecast" was false for the regime half.** Lever 1 was costed as
saving ~876s, of which ~183s came from `forecaster.py:4210` skipping regime models whenever HP
was reused. But `predict` at `:5584` **prefers the regime model over the global one** whenever
`_detect_current_regime` matches, so an artifact with no regime models serves a different mid.
Dropping them is a change to the forecast, not a cost saving.

That was tolerable while warm retrains only ran locally. Restoring the cache in CI makes the
warm path *production's steady state*, so the coupling had to go: `SKIP_REGIMES=1` is now the
only way to skip them. **The lever is therefore worth ~693s, not ~876s.** The regime branch
also now clears the horizon's restored regime models before refitting — otherwise a regime that
falls below `MIN_REGIME_TRAIN` is `continue`d and its *previous* run's model survives to be
re-persisted beside freshly trained global models.

**2. Optuna's bimodality is not the pruner.** The 2026-08-10 retrain note attributes the
63.3s → 692.9s swing to "the pruner killed most trials at iteration 5-15". There is no pruner:
`forecaster.py:2950` states outright "no early stopping, no pruner", and explains why
(`LightGBMPruningCallback` prunes on LightGBM's `quantile` metric, not on the within-date rank
IC this objective returns). Both runs also used `TPESampler(seed=42)` with ≤15 trials, and
TPE's first 10 are seeded startup draws — so **both runs searched identical hyperparameters at
identical round counts.** The swing is data-dependent tree growth: `31337078991` ran on
pre-`873148b` labels, `31356483719` on the post-re-vote ones. Conformal CV moved inversely
(836.5s → 514.1s) across the same pair.

Consequence for budgeting: the expensive draw is the one on *current* labels, so **quote ~693s
saved, not "246–876s, one draw of a wide distribution."** Per-horizon Optuna this run was
0.0 / 198.0 / 68.0 / 426.9s.

**3. The ~2GB parquet is written on the predict path, not the training run.** The doc's lever 4
says "CI writes a ~2GB parquet every training run and discards it". `_save_engineered_cache`
has exactly one caller — `predict` at `:5483` — so the waste is on the *daily* run. Feature
engineering still has to run either way (the cache is excluded from the restore, so it always
misses); only the serialization is saved. Still worth doing, smaller than implied, and on a
different leg than described.

## Not changed

`HORIZON_EXCLUDED_GROUPS` is also bypassed on a warm retrain (`:4578`), and was left alone:
`FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` already excludes `cross_sectional` and
`events`, so the branch is a no-op on the shipped configuration. Feature-group validation stays
skipped on warm retrains — it is a diagnostic and a pruning trigger, not a served artifact.

## Verification

`tests/test_minimal_model_shape.py` +5 tests; full backend suite green. Two of the new tests
parse the workflow, so the restore gate and `FORCE_RETRAIN` cannot drift apart.
`test_warm_retrain_still_trains_regime_models` was confirmed to **fail** against the old
coupling rather than pass vacuously.

**Still unverified:** whether HP reused across a label change is as good as a fresh search. The
cost is now measured; the *quality* of reuse is not. Re-tune periodically with
`FORCE_HP_SEARCH=1` — a label-basis change like `873148b` is exactly when cached params go
stale, and nothing detects that automatically.

## Next levers, unchanged in rank

Split the conformal CV's two masters (~700s of ~514–836s, calibration split carved out
*before* HP selection), then the 4-way horizon matrix (wall clock only). Free and untried: the
arm64 runner — `ubuntu-24.04-arm` is free on public repos and ~40% faster, needs an aarch64
wheel check for lightgbm/duckdb/onnxruntime. Do **not** cut boost rounds (30d PT margin is
t=3.35 against the t=3.06 the count was calibrated for), `CV_STEP_DAYS` (folds are PT's sample
size; 14d is marginal at t=3.077), or `TRAIN_FEATURE_ROWS` (re-engages the item-draw subsample
the 1.2M budget exists to avoid).
