# arm64 runners, and a scheduled job that scores the signal production serves

**Date:** 2026-08-10
**Follows:** `docs/changelog/2026-08-10-warm-retrain-in-ci.md` (HP reuse, ~693s).

## 1. `runs-on: ubuntu-24.04-arm`

Free on public repos and materially faster on a LightGBM-bound workload. Every binary
dependency ships an aarch64 manylinux wheel, verified against PyPI before switching —
`lightgbm` as `py3-none-manylinux2014_aarch64`, the rest as cp311 builds — so nothing compiles
from source. No code change.

**Compare within an architecture, never across.** Floating point differs, so an artifact
trained here is not bit-identical to an x86 one and any delta measured across the switch
confounds architecture with everything else. The x86 baseline is run `31405829674`, dispatched
before this change and on the parent commit.

## 2. `model-diagnostics.yml` — weekly, one job per horizon

`price-forecast.yml` sets `CV_DIAGNOSTIC_CLASSIFIER=0` because the per-fold classifier is 52% of
a retrain. The consequence is the one that matters: `classifier_accuracy` is `None` on every
fold and `served_acc` falls back to the q50 sign, so **every DA and rank-IC number in a daily
log describes the quantile sign, not what production serves.**

That is why the naive-mid question cannot currently be decided. The q50 loses to `−return_1d` on
rank IC at all four horizons, which is a ~60%-of-training saving if acted on — but the
comparison is between two candidates for the served mid, judged by a metric that does not
describe the served mid. This job buys the measurement back on a schedule rather than on the
daily path.

Four guards keep a measurement from becoming a deployment:

- **No cache save.** There is a restore step and deliberately no save. The `forecast-models-`
  key is the *only* route to production — the daily predict run restores the newest entry by
  prefix — so saving a single-horizon artifact there would silently promote it. Pinned by
  `test_diagnostics_workflow_cannot_promote_its_artifact`.
- **`FORECAST_MODEL_DIR` deliberately unset.** Pointing it at a scratch directory looks safer
  and is worse: `load_models()` would read an empty directory, `tuned_params` would stay `{}`,
  Optuna would re-search, and the job would measure a *freshly tuned* model instead of the one
  production trains. Also pinned.
- **`TRAIN_HORIZONS`** (new, `scripts/forecast_prices.py`) gives one horizon per matrix job. The
  horizon-matrix decomposition is risky in `price-forecast.yml` because a partial artifact must
  be merged exactly; here the artifacts are discarded, so that hazard does not exist. An
  unparseable or unknown value trains **every** horizon — same rule as
  `TRAIN_MIN_MEDIAN_PRICE`, because a partial artifact that looks complete is the dangerous
  outcome.
- **`SKIP_REGIMES=1`.** Unlike on the serving path, this costs no accuracy here: nothing this
  job produces is predicted from.

## The conformal split is NOT done, and the plan changed

Deliberately left open. The research doc costs it at ~700s by replacing 33 CV fits with one
held-out calibration fit per horizon, and labels it "no effect on the served point forecast".
The point forecast, correct. **The band is a different question.**

A single trailing calibration window carries ~30 distinct dates, and within a date the whole
cross-section moves together — so its **effective** sample size is the date count, not the row
count. That is precisely the pathology recorded in `early-stopping-destroys-signal`: the
trailing val window has "~a dozen effective observations", which is why early stopping halted
at round 1. The 8-fold OOF set spans **eight different market periods**; one window spans one.
Swapping them trades multi-period coverage robustness for speed, and coverage is a served
property.

**Revised plan, which this workflow unlocks.** Keep CV for calibration but cut its fold count
for the *daily* run, and let the scheduled job run the full fold count for PT and rank IC. The
doc rejected `CV_STEP_DAYS` 150→300 on one ground only — "fold count is also the sample behind
PT, whose t scales with √n" — and that objection dissolves once PT lives on a schedule. Halving
folds saves ~50% of the CV phase (~250–400s) while keeping q_hat multi-period.

**Not shipped, on purpose:** it changes band coverage, and there is no coverage measurement in
hand. Shipping a coverage change blind is the exact failure this session has been correcting.
Land the diagnostics job, read coverage, then cut folds.

## Verification

Full backend suite green; +9 tests in `tests/test_minimal_model_shape.py` across both changes.
Both workflow files parse. **Nothing here is measured yet** — the arm64 speedup is an estimate,
and `model-diagnostics.yml` has never run.
