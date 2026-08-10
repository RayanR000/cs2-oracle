# arm64 runners, and a scheduled job that scores the signal production serves

**Date:** 2026-08-10
**Follows:** `docs/changelog/2026-08-10-warm-retrain-in-ci.md` (HP reuse, ~693s).

## 1. `runs-on: ubuntu-24.04-arm` — measured at ~11%, not the 25–40% advertised

Free on public repos. Every binary dependency ships an aarch64 manylinux wheel, verified against
PyPI before switching — `lightgbm` as `py3-none-manylinux2014_aarch64`, the rest as cp311 builds
— so nothing compiles from source. No code change.

**Measured: run `31407938154` (arm64) against `31405829674` (x86).** Both `train-only`, both warm
on HP and on the `voted-v6-` frame, identical universe, identical round counts — architecture is
the only variable, so unlike the previous comparison this one is clean.

| Phase | x86 `31405829674` | arm64 `31407938154` | Δ |
|---|---|---|---|
| q50 ensemble | 232.2s | 232.4s | **+0.1%** |
| Direction classifier | 214.1s | 135.4s | **−36.8%** |
| Conformal CV | 392.8s | 360.2s | −8.3% |
| Remainder | ~278.8s | ~268.6s | −3.7% |
| **TOTAL training** | **1117.9s** | **996.6s** | **−10.8%** |
| Job | 19m52s | **17m48s** | −10.4% |

**The estimate published before the run said 25–35%, quoting GitHub's "up to 40%". That was
wrong — it is ~11%.** Worth keeping (free, ~2 minutes off every run, no code change) but not a
lever to plan around.

**The gain is concentrated, not uniform.** 65% of it is the direction classifier alone. The q50
ensemble is flat to 0.1% — the quantile objective sees no benefit. That flat phase doubles as a
control: two runs doing identical work at identical cost on one phase is what makes the
classifier's −36.8% credible rather than runner noise. Caveat that this is n=1 per arm against a
±25% per-phase swing, so treat ~11% as indicative; only the classifier delta is clearly outside
noise.

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

Full backend suite green (1769); +9 tests in `tests/test_minimal_model_shape.py` across both
changes. Both workflow files parse. arm64 is measured (above). **`model-diagnostics.yml` has
never run** — its ~16 min/horizon estimate is unverified, and so is the assumption that a
classifier-on CV fits the cap once decomposed.

## Cumulative, across both changelogs

| | Training | Job |
|---|---|---|
| `31356483719` (cold HP, cold vote, x86) | 2306.0s | 39m32s |
| `31405829674` (warm HP, warm vote, x86) | 1117.9s | 19m52s |
| `31407938154` (warm HP, warm vote, arm64) | **996.6s** | **17m48s** |

**−57% training, −55% wall clock**, with −692.9s of it attributable to HP reuse and −121.3s to
arm64. The rest is a warm voted cache and per-phase variance, not a change anyone made.
