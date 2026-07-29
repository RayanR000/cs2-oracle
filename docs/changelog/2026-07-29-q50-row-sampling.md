# q50 row sampling: GOSS → bagging (shipped, 2026-07-29)

## The defect

`data_sample_strategy="goss"` was applied to the **q50 quantile only**; q10/q90
used bagging. GOSS subsamples rows by ranking them on |gradient| — but the
quantile objective emits **constant ±alpha gradients**, so the ranking is
degenerate (every row ties) and the compensating rescaling of the
small-gradient set injects bias rather than signal.

Symptom in the shipped artifacts (retrain 2026-07-27, inspected 2026-07-29 —
LightGBM saves up to `best_iteration`, so tree count = rounds that helped):

| Horizon | q10 | **q50** | q90 |
|---|---|---|---|
| 3d | 11 / 1 / 5 | **1 / 22 / 5** | 121 / 64 / 89 |
| 7d | 114 / 49 / 56 | **1 / 1 / 2** | 59 / 47 / 52 |
| 14d / 30d (DART) | 500 | 500 | 500 |

7d q50 was a 1-tree model across all three ensemble members — the served
median forecast was effectively an intercept.

> **`docs/architecture/model-optimization.md` had this backwards.** It recorded
> "p90 GBDT is broken (1-3 rounds, GOSS incompatibility)". p90 is the
> *healthiest* GBDT quantile; q50 was the broken one. Corrected in that doc.

## Root-cause evidence (before the A/B)

Controlled repro on the full production frame (110K rows, 48 features,
production split and params), varying only the sampling strategy:

| Cell | 3d best_iter / pinball | 7d best_iter / pinball |
|---|---|---|
| production (GOSS) | 113 / 5.7783 | 12 / 7.2205 |
| bagging | 266 / 5.7281 | 163 / 7.0873 |
| bagging + `min_gain_to_split=0` | 523 / 5.6797 | 97 / 7.0680 |

`min_gain_to_split=0.1` was **ruled out** as a contributor — GOSS with
gain=0 was identical to production to six decimals.

## A/B (`scripts/ab_test_q50_sampling.py`)

Two arms, identical folds / data / tree params (pulled from the persisted
`meta.json:tuned_params`), differing only in `data_sample_strategy`.
Production purge-gap CV (`_compute_cv_splits(purge_days=horizon)`), production
sample weights, 8 folds, 200 items, one ensemble member at
`feature_fraction=0.7`.

**Pre-registered gate** (fixed before running): ship iff mean paired pinball
improvement ≥ 0.5% relative AND bagging wins in ≥ half the paired folds AND
directional accuracy does not regress > 0.5pp. Pinball is primary (it is the
loss q50 optimizes); DA is a guard only, since the served up/flat/down signal
has come from the directional classifier since 2026-07-24.

### Result — SHIP on both GBDT horizons

| Horizon | Arm | Pinball | MAE | DA | Avg trees |
|---|---|---|---|---|---|
| 3d | goss | 2.49898 | 4.9980 | 53.85% | 727.5 |
| 3d | **bagging** | **2.37694** | **4.7539** | **54.97%** | 904.9 |
| 7d | goss | 3.15157 | 6.3031 | 55.46% | 135.1 |
| 7d | **bagging** | **3.11742** | **6.2348** | **56.17%** | 235.5 |

| Gate | 3d | 7d |
|---|---|---|
| Pinball ≥ +0.5% | +5.21% PASS | +1.76% PASS |
| Folds won ≥ half | 7/8 PASS | 5/8 PASS |
| DA regression ≤ 0.5pp | +1.13pp PASS | +0.71pp PASS |

7d passes thinly (5/8 folds, +1.8%); 3d is the strong result. Per-fold spread
is wide on 7d (−8.96% to +10.93%), so treat the 7d effect as small-but-real.

**Two distinct harms, same root cause.** On 7d, GOSS stunted training
(best_iteration ≈ 1). On 3d, goss often ran the full 1000 rounds and bagging
*still* won — fold 0 had both arms at 1000 trees with bagging 6.85% ahead. So
GOSS also biases the fit when training runs to completion.

## Change

Both param paths now route through one helper pair, so the three former
call sites cannot drift:

- `_row_sampling_params(quantile, subsample=0.8)` — bagging for every
  quantile; sets `bagging_freq=1` for q50.
- `_apply_row_sampling(params, quantile, subsample=None)` — strips every key in
  `_ROW_SAMPLING_KEYS` before rewriting, so GOSS keys cached in a
  pre-2026-07-29 `meta.json` cannot leak through the warm-retrain path.

Call sites updated: the Optuna objective (q50 now searches `subsample` like
q10/q90), the warm-retrain reuse path, and the cold `base_params_by_q` path.

Tests: `TestQ50RowSampling` (7 tests) in `tests/test_forecaster.py`.

## Follow-ups (NOT shipped here)

1. **`bagging_freq` was never set anywhere before this change.** LightGBM
   defaults it to 0, which ignores `bagging_fraction`/`subsample` entirely —
   so q10/q90's `subsample=0.8` has always been a **no-op**, and Optuna has
   been searching a dead parameter for them. This change sets `bagging_freq=1`
   for **q50 only**, because that is the exact config the A/B validated.
   Correcting q10/q90 is a real change that needs its own A/B — it would alter
   interval width, and interval coverage is already poor (39–48% against an
   80% target). Do not "tidy" the asymmetry away without measuring it.
2. **14d/30d not evaluated.** DART trains a fixed 500 rounds with no early
   stopping, so the collapse mode cannot occur there; only the bias mode could.
   The harness shards them: `--horizon 14 --fold-start 0 --fold-count 4`.
3. **A confirming production retrain is required.** The A/B ran on 200 items
   with a single ensemble member; it does not establish the effect on the full
   100K-row frame with the 3-member ensemble. Verify with
   `scripts/backtest_accuracy.py` against the DA floors in
   `docs/architecture/model-optimization.md`.

## Incident: the test suite overwrote production model artifacts

Discovered while verifying this change. `test_regime_models_populated_after_train`
constructed `ItemForecaster(db_session=MagicMock())` with **no `model_dir`**, so
it defaulted to the real `models/saved_models/`, and `train()` persists at the
end — overwriting **132 of 142 files** with models fit on 5 synthetic items.
`*.txt` artifacts are gitignored and therefore unrecoverable; `meta.json` is
tracked and was restored from HEAD (the 2026-07-27 retrain, 9 CV folds,
conformal calibration on all 4 horizons).

Fixed on both levels:
- That test now takes `tmp_path` and passes `model_dir=str(tmp_path)`.
- The shared `forecaster` fixture now defaults to
  `tmp_path_factory.mktemp("saved_models")`, so no future test can write to the
  real directory by omission.

Verified: a full `tests/test_forecaster.py` run now modifies **0** files in
`models/saved_models/`.

**Consequence: the on-disk `lgb_*.txt` / `clf_*.txt` artifacts are currently
models trained on synthetic test data and must not be served.** A retrain is
required regardless of this change.
