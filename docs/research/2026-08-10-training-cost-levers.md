# Training cost: what is left, and what it would cost in accuracy

**Measured 2026-08-10** against CI runs `31337078991` (1884.2s training, 35m24s job) and
`31356483719` (2306.0s, 39m32s). Supersedes the cost accounting in
`docs/changelog/2026-08-09-training-cost-levers.md`, whose Phase 1 levers all landed and which
concluded — correctly — "do not quote a speedup from this work".

The project's wall-clock cap is 30 minutes per run. Both runs above are over it.

> **Corrected 2026-08-10 when levers 1 and 4 landed.** Three claims below are wrong; see
> `docs/changelog/2026-08-10-warm-retrain-in-ci.md`. (a) Lever 1 is worth **~693s, not
> 246–876s**, and it is **not** free of served effects: the ~183s regime-model half changes the
> served mid, because `predict` prefers regime models over the global one. (b) Optuna's
> bimodality is **not** the pruner — there isn't one — so the expensive draw is the one on
> current labels and is what to budget from. (c) Lever 4's parquet is written on the **predict**
> path, not the training run. Lever 1 also required `FORCE_RETRAIN=1` on `full`, which this doc
> does not mention.

## Where the time goes

Per-horizon wall clock from run `31337078991`, timestamps not `[timing]` lines, so it includes the
un-instrumented phases:

| | 3d | 7d | 14d | 30d | shared |
|---|---|---|---|---|---|
| wall clock | 305s | 532s | 137s | **751s** | 150s data build |

**30d is ~40% of the run, 14d is 7%.** The two instrumented phases that dominate — Optuna and
conformal CV — are also the two that swing hardest between runs (Optuna 63.3s → 692.9s; conformal
CV 836.5s → 514.1s across the two runs above). Any lever quoted from a single run is quoted from
one draw of a wide distribution.

## The levers, ranked

### 1. Let training runs restore the model cache — free, and it removes the worst variance

`price-forecast.yml`'s "Restore trained models" step is gated `if: mode == 'predict-only'`. So a
`full` or `train-only` run begins with an empty `saved_models/`, `load_models()` returns False,
and `self.tuned_params` stays `{}` — it is populated only by the meta.json load path at
`forecaster.py:6632-6645`. Two consequences:

- **Optuna re-searches from scratch every training run.** 63.3s in one run, **692.9s** in the
  next. This is the single largest and least predictable phase.
- **Regime models retrain.** `forecaster.py:4210` skips them whenever HP was reused
  (`_warm_retrain`), so restoring the cache picks up ~183s without needing `SKIP_REGIMES=1`.

**Saves 246s in the cheap case, ~876s in the expensive one.** No effect on the served forecast —
it is the warm path that already runs locally. Upkeep: re-tune periodically with
`FORCE_HP_SEARCH=1` so the cached hyperparameters do not go stale indefinitely.

### 2. Run the four horizons as parallel matrix jobs — wall clock only

The horizon loop at `forecaster.py:3970` is sequential and the horizons are independent. Every
piece of per-horizon state is dict-keyed by horizon (`models`, `direction_models`, `regime_models`,
`conformal_calibration`, `tuned_params`, `cv_results`); the shared values (`feature_cols`,
`feature_medians`, `sigma_clip`, regime thresholds) are frame-level and identical in each branch.

As a 4-way matrix the run is bounded by 30d: ~751s + 150s data build + ~90s setup ≈ **15 minutes**,
from 35. Zero modelling change.

**Cost:** each job repeats the data build, so total compute rises while wall clock falls — the cap
measures wall clock. Requires a partial-artifact merge step, which must be exact or the run ships
a half-model; that wants its own test. The `ThreadPoolExecutor` deadlock recorded at
`forecaster.py:4105` does not apply — that was nested thread pools inside one process.

### 3. Split the conformal CV's two jobs apart — ~700s

Conformal CV is 44% of run `31337078991` (836.5s) and refits a median model per fold per horizon —
33 fits — to produce out-of-fold residuals for `q_hat`.

`forecaster.py:4274-4292` explains why it cannot simply be skipped, and the reasoning is sound:
`X_val` is the set `lgb.early_stopping(50)` stopped on *and* the set `_optuna_search_params` scores
against, so a `q_hat` fitted there is measured on rows the model was selected against, comes out
biased low, and the served band under-covers. `SKIP_CV` was removed from CI for exactly that
reason.

**But early stopping was removed on 2026-08-09** by lever 1.2 (`FIXED_BOOST_ROUNDS`), so half that
justification is gone, and on a warm retrain Optuna does not run either.

The structural observation: **the CV serves two masters** — calibrating `q_hat`, and computing the
rank-IC and PT diagnostics — **and only one needs to run on every retrain.** A calibration split
held out from both training and HP selection is one extra fit per horizon (~100s total) rather
than 33. The full expanding-window CV then runs on a schedule for the PT verdict and the rank-IC
series.

**Saves ~700s. No effect on the served point forecast.** Design caveat: on a warm retrain the
cached hyperparameters were selected against those rows in an *earlier* run, so the calibration
split has to be carved out before HP selection, not after — otherwise this recreates the exact
under-coverage bug the comment warns about.

### 4. Stop writing `engineered_data.parquet` in CI — unmeasured

`price-forecast.yml:93,238` exclude it from both cache save and restore, so CI writes a ~2GB
parquet every training run and discards it. Useless in CI specifically; it is a real cache locally.

## Levers that are not levers

- **Boost rounds.** Calibrated 2026-08-08 at the knee (`forecaster.py:577-628`), with pinball loss
  and rank IC documented as pointing in opposite directions at 14d/30d. 30d sits deliberately at
  the peak: at 500 CV rounds its PT drops t=3.06 → 2.81, "no_skill". The 2026-08-10 retrain
  measured t=3.35, so the margin is real but not large. **Cutting rounds is the change that looks
  obviously right and would quietly cost you the horizon.**
- **`CV_MAX_TRAIN_ROWS`** already binds at exactly 300,000 on all four horizons.
- **`CV_STEP_DAYS` 150 → 300** halves the conformal phase, but fold count is also the sample
  behind PT, whose t scales with √n. 14d is now at t=3.077. Free on accuracy, not free on the
  ability to *claim* accuracy.
- **`max_bin`** is at 63; lowering it to 31 measured *slower* (26.3ms vs 25.5ms per round).
- **`N_ENSEMBLES = 1`** — the ensemble loop is a loop of one. **`QUANTILES = [0.5]`** — the p10/p90
  GBMs are long gone. Feature engineering is 12s. No GPU on the runners.

## The structural option

The q50 model loses to `−return_1d` on rank IC at all four horizons (2026-08-10: −0.0159 /
−0.0371 / −0.0433 / −0.0091), and none of those gaps is significant — the ML price model is
statistically indistinguishable from a one-line baseline.

If the served mid were the naive predictor, the q50 ensemble (~307s) disappears and calibration
becomes nearly free, since residuals of a closed-form predictor need no model fit. That is ~1,100s,
**about 60% of training**, and measured rank IC would rise. The served direction comes from the
classifier and would be untouched.

Recorded, not recommended: it is a decision about what the product is, not a cost tweak.

## Budget

| Change | Saves | Accuracy effect |
|---|---|---|
| 1. Restore model cache on training runs | 246–876s | none |
| 3. Split-conformal calibration, scheduled diagnostics | ~700s | none served; PT published on a schedule |
| 2. Then parallelize horizons | wall clock only | none |

Levers 1 and 3 take training from 1884s to roughly 940s (~16 min job). Adding the matrix leaves the
30d branch at ~290s of work, landing the job near **9–10 minutes**.

## Two facts worth carrying forward

**The served signal is unscored in CV.** `CV_DIAGNOSTIC_CLASSIFIER=0`, so `classifier_accuracy` is
`None` on every fold and `served_acc` falls back to the q50 sign. Every DA number in a CV log
describes the quantile sign, not what production serves.

**For 7d, 14d and 30d the last CV fold ended 2026-02-09** on the 2026-08-09 artifact — only 3d
reached July. `q_hat`, rank IC and PT for three of four horizons were calibrated on data ending
before the 2026-03-22 consensus break and every July cutover. Not a cost issue; flagged because it
bounds what those numbers mean.
