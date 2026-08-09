# Model Optimization Options

> Covers the levers that make the model smaller, faster, and cheaper to train/infer.
> Source of truth: `backend/models/forecaster.py:ItemForecaster` class constants.

> **⚠ The accuracy side of every trade-off below is unverified.** The timing measurements are
> sound. The accuracy costs are not: they were measured either on the served series (effective
> sample size 1–2 forecast dates) or on the feature A/B harness, whose noise floor is ~1.15pp at
> 3d and up to 7.13pp at 30d — wider than most of the per-lever deltas quoted. Run
> `scripts/compute_mde.py` before treating any sub-1pp figure as real. See
> `docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md` and
> `docs/research/accuracy-opportunities.md`.

---

## Current Baseline (2026-08-05)

| Metric | Value |
|--------|-------|
| **Models** | **8** — 4 median GBMs (4H × 1Q × 1E) + 4 directional classifiers. ⚠️ **Was 36 global (4H × 3Q × 3E) + ≤108 regime, i.e. 40 files on disk.** Collapsed 2026-08-05 by the minimal-model rewrite (`1902aab`); see `docs/changelog/2026-08-04-minimal-model-results.md`. Regime models are still supported by the code and would be 3R × 4H × 1Q × 1E = ≤12; `predict()` falls back to global cleanly and logs `no regime models trained, using global` |
| **Ensemble size** | `N_ENSEMBLES = 1` (seed 42; feature fraction 0.7). ⚠️ **Was 3** (seeds 42, 73, 91; fractions 0.6, 0.7, 0.8). The 3→1 collapse is **unmeasured** — the walkforward gate never ensembled, so it cannot resolve it. Argued safe for served DA on structural grounds only (the served direction comes from the classifier, which was never ensembled). If accuracy needs recovering, `N_ENSEMBLES = 2` is the first thing to try |
| **Horizons** | 3d, 7d, 14d, 30d — **all GBDT**. ⚠️ **14d and 30d were DART.** Removing DART was measured as an *improvement* at 14d (+3.14pp paired DA, CI [+1.96, +4.37]) and neutral at 30d, against the spec's expectation that it was the riskiest change |
| **Quantiles** | **p50 (0.5) only.** ⚠️ **Was p10/p50/p90.** The band now comes from split conformal around the single median model (`models/conformal.py`), not from quantile models: 24 p10/p90 GBMs cost 223.2s of a 381.2s budget for 39–48% empirical coverage against an 80% target. Retained for the record: an earlier version of this row read "p90 GBDT is broken (1-3 rounds, GOSS incompatibility)" — that was backwards. p90 was the healthiest GBDT quantile; **q50** was the collapsed one (7d q50 saved 1–2 trees), because GOSS was applied to q50 only and is degenerate under the quantile objective's constant ±alpha gradients. Fixed 2026-07-29 — see `docs/changelog/2026-07-29-q50-row-sampling.md` |
| **Boost rounds** | 1000, early stop 50 — unconditionally. ⚠️ **`DART_NUM_BOOST_ROUND = 500` and `BOOSTING_TYPE_MAP` are both deleted** (2026-08-05), along with every `!= "dart"` branch they selected; `BOOSTING_TYPE = "gbdt"` is now a scalar. The map was never mutated by anything, so DART was already unreachable — the only remaining hard-coded user was `scripts/ab_test_hp_search.py`, itself a shelved experiment, deleted with it. `tests/test_minimal_model_shape.py::test_dart_is_gone_from_the_forecaster` guards against re-entry |
| **Features** | **47 columns** in the shipped 2026-08-05 artifact, identical across all four horizons (`meta.json: horizon_feature_cols`). That is the count *after* correlation pruning at 0.95, `SHELVED_FEATURES`, and `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]`; every other feature group is engineered on every training row and then discarded, and the allowlist logs the drop as `pre -> post`. ⚠️ **An earlier version of this row read "~70–120 features, 8 groups"; that was the engineered count, not the served count, and the doc then contradicted itself further down.** Shelving the 13 volume features (their archive column has been identically 0 since 2026-05) took the served count to **36** in the 2026-08-06 artifact, and shelving the 37 dollar-denominated columns takes it to **32** on the next retrain (training smoke run, 2026-08-06 — see `docs/changelog/2026-08-06-scale-free-features-and-fabricated-labels.md`). `MODEL_ARTIFACT_VERSION` is now **5**; each bump exists to force the retrain rather than wait out the 14-day age trigger |
| **Rows** | `max_feature_rows = 100_000`, overridable with **`TRAIN_FEATURE_ROWS`**. This is an item-coverage budget, not a row budget: it selects **99 whole item histories of ~5,377** (1.8% of the pool). See the dedicated section below |
| **HP search** | 3d skipped (frozen 50-trial winner, warm-started in `_optuna_search_params`), 7d=10 trials, 14d=15, 30d=15. Cold Optuna across all four horizons measured **39.4s** total and a warm retrain logs 0.0s, so HP search is no longer a meaningful cost centre |
| **Warm retrain** | **176.7s** measured 2026-08-05 (10-core Mac, `SKIP_REGIMES=1 --train-only`, HP cached). ⚠️ **Was 381.2s** on the same machine and flags. Booster fitting fell 330.8s → 28.1s, but the out-of-fold conformal CV rose 50.4s → ~148.6s and was then **~84% of training**, so the bottleneck is calibration, not the model. ⚠️ **These are 100K-config numbers.** The shipped ≥$1 fixed-rounds config is an **872s** cold retrain in which CV is 439.3s / 50.4% — see "Where the time goes now" |
| **Cold retrain** | **250.1s** measured 2026-08-05 (same flags, no cached HP; Optuna 39.4s). ⚠️ **Was 12m30s** measured 2026-07-29. The first post-rewrite run is necessarily cold: the artifact-version check refuses the pre-rewrite `meta.json` |
| **Inference** | **87s measured locally on a cold cache** (2026-08-04, `VOTED_CACHE=0`, 10-core Mac, 8,691 items / 34,764 forecasts). Phases: 37s fetch (730d window, 5.37M raw → 3.59M voted), 1.4s tail (3.59M → **1.39M rows**), 35s chunked feature engineering, remainder booster scoring. **72 of those 87s are fetch + feature engineering, so predict is data-bound, not model-bound** — scoring 4 boosters instead of 36 shrinks only the remainder, and this figure was not expected to move on the collapse. Not re-measured on a 2-core CI runner, where the local number does not transfer |
| **Production DA** | **Not quotable.** `MIN_FORECAST_DATES = 20` (`backtest/scoring.py:38`) and every live cohort spans 1–2 distinct forecast dates, so all four horizons report NO HEADLINE. It also cannot be refreshed on demand: `backtest_accuracy.py` scores *stored* forecasts against matured actuals and maturity is bounded by archive coverage, so no post-rewrite forecast has matured. ⚠️ **Every production DA measured before 2026-08-02 is an artifact** of a scorer whose two legs used different estimators — one 5,512-forecast cohort scored 61.76%, 33.74%, 61.54% and 57.91% on four evaluation dates with no new data. Do not cite those figures and do not read their disappearance as a model regression. See `docs/changelog/2026-08-01-deterministic-backtest.md`. For a fresh-model number, run `scripts/walkforward_backtest.py` |
| **Classifier CV DA** | **3d=67.3%, 7d=67.8%, 14d=68.1%, 30d=67.9%** (9 folds; 8 for 30d) — measured during the 2026-08-05 post-rewrite retrain. This is the **served** direction signal (classifier, not quantile-sign, since 2026-07-24) but a training-time diagnostic only: interior rows, a 99-item universe, ~83% tier-0, so the all-tiers number reads close to a penny-item score. The ≥$1 cohort is now reported alongside it (2026-08-05). **Not** comparable to a production figure |
| **Quantile-sign CV DA** | 3d=61.0%, 7d=61.0%, 14d=60.8%, 30d=64.0% (same run; sd 5.2–8.9%). ⚠️ **Understates the shipped model:** the CV block caps fits at `num_boost_round=200`, but production 3d q50 trains to 349–428 rounds — CV scores an under-trained model. Does not affect the classifier row (different code path) |
| **7d q50 early stopping** | ⚠️ **Noise-determined.** The val curve improves only 0.28% total on the production frame, so `early_stopping(50)` trips on noise: stopping round sd 41 on mean 47 (shipped members 41/11/89). Root-caused 2026-07-29 to a regime mismatch — the 30-day val window has ~2× the return spread of the training set, and only 14.5% of training rows fall in the last 180 days. Tree count is a red herring: 3d looks healthy only because its `learning_rate` is 0.01 vs 7d's 0.068. **Any 7d accuracy A/B is partly measuring this noise.** See `docs/changelog/2026-07-29-7d-q50-early-stop.md` |

---

## Where the time goes now

⚠️ **The table below was re-measured 2026-08-09 on the shipped fixed-rounds config**
(`FIXED_BOOST_ROUNDS` = 3d 300 / 7d 750 / 14d 150 / 30d 1000, verified from the saved boosters'
tree counts), a CI-exact cold retrain on an uncontended 10-core Mac: **872s training + ~3s data
build** (warm voted cache). An earlier version of this section read **487s** with conformal CV at
260.8s / 53.6% — that was the early-stopping config, which was cheap by not training. Source:
`docs/changelog/2026-08-09-shipped-retrain-cost-measured.md`.

| Phase | Seconds | Share |
|---|---|---|
| Conformal CV (33 folds × 4 horizons) | 439.3s | **50.4%** |
| q50 ensemble | 158.4s | 18.2% |
| Regime models | 95.4s | 10.9% |
| Direction classifier | 92.3s | 10.6% |
| Remainder (targets, splits, feature medians, artifact save) | 52.0s | 6.0% |
| Optuna | 35.0s | 4.0% |

⚠️ **Read the total, not the phases.** Per-phase timings on this machine swing ±25% between clean
runs at identical rounds — 3d conformal CV read 93.9s and 68.3s across two uncontended runs of the
same config. Do not compare a single phase across runs, and do not size a lever from one.

**The CV diagnostic classifier is 52% of a research retrain, and CI already gates it off.** The CV
phase fits *two* boosters per fold: only the q50 predictions become `oof_records` → conformal
`q_hat` and the confidence thresholds. The 3-class direction classifier fitted beside it feeds only
`classifier_accuracy` / `classifier_accuracy_ge1` in `cv_results`, which is serialised to
`meta.json` and read by no served artifact. Measured 2026-08-09 by running the shipped config both
ways: **872s with `CV_DIAGNOSTIC_CLASSIFIER=0` (what CI runs) against 1804s with it on** — the
diagnostic costs **932s, 52% of a classifier-on retrain**. ⚠️ **An earlier version of this
paragraph said 69% of a fold's fit cost, ~155–180s, 32–37% of the retrain.** That was benchmarked
under early stopping, which collapsed the classifier to ~3 trees; fixed rounds let it train
full-length at 3 trees per round, so its share roughly doubled. The local/research default is
still on, which puts a research retrain at 30.1 min.

**This inverts the pre-rewrite picture, and it is what makes most of the old lever tables moot.**
Before the collapse, two DART horizons were 78% of training and the 24 p10/p90 boosters were 59%;
both are gone. Every surviving lever that acts on *booster fitting* is now competing for the three
fit phases above (130.5s combined), which is why the sub-1pp micro-levers below are no longer worth
a retrain individually. The phases with real headroom left are fold count and the per-fold
diagnostic classifier.

**Feature engineering is 7s, not 35s**, at this config — see lever 2, which is sized against it.

**`n_jobs = max(1, cpu_count // 2)`** (`forecaster.py:3866`) was vestigial from the
parallel-ensemble code deleted 2026-07-21, and contradicted the comment above it; it has now been
changed to `-1`. Expect less than it looks: LightGBM is memory-bandwidth bound at this frame shape,
and 1 → 10 threads measured only **1.55×**, so thread count is worth ~25% of the booster-fit
phases, not 2×.

The pre-rewrite per-horizon/per-quantile breakdown lives in
`docs/changelog/2026-08-04-minimal-model-results.md`. It is not reproduced here: the models it
measures no longer exist.

---

## The training row budget

The knob that was missing from this document entirely, and the largest single decision in it.

`TRAIN_FEATURE_ROWS` (env) → `DEFAULT_TRAIN_FEATURE_ROWS = 100_000`
(`scripts/forecast_prices.py:45`, parsed at :54, passed to `train(max_feature_rows=...)` at :265).
An unparseable or non-positive value logs a warning and falls back to the default. Distinct from
`TRAIN_HORIZON_MAX_ROWS = 700_000` (:50), which caps each horizon's slice *after* feature
engineering and is not the coverage dial — the two are deliberately not unified, because one number
would otherwise move coverage and the per-horizon cap at the same time.

`_stratified_item_subsample()` (`forecaster.py:2419`, applied at :2496) spends the budget on
**whole item histories** — stratified by rarity, full calendar window preserved, so lags and
rolling features stay valid. It therefore decides how much of the item universe the model ever
sees:

| `TRAIN_FEATURE_ROWS` | Items selected | Training time | Measured |
|---|---|---|---|
| 100,000 (default) | 99 of ~5,377 (**1.8%**) | **104.6s** | 2026-08-05 |
| 700,000 | 646 (12%) | **468.7s** | 2026-08-05 |
| 1,200,000 **with `TRAIN_MIN_MEDIAN_PRICE=1.0`** | 926 of 5,542 — the whole ≥$1 cohort, no subsample | **538s** | 2026-08-07 |

Raising it is a ~4.5× cost increase — more than the pre-rewrite 40-model grid cost — which spends
the entire minimal-model saving to buy 12% of the pool, and **the fresh-model gate cannot detect
the resulting accuracy difference**. So there is no measurement that would justify the spend, and
it was explicitly declined; the rationale is in the code at `forecaster.py:2791-2806`.

**Two 2026-08-07 corrections to how that cost should be read.** First, the retrain runs on
**Mondays only** (`price-forecast.yml`, `date +%u = 1`); every other day is predict-only, so a
budget increase is a weekly cost, not a daily one. Second, the third row above is the cheaper way
to buy coverage: the median-price floor drops the 82.56% of item-days that are sub-$1 *before* the
budget is spent, so 538s buys the entire served cohort rather than 12% of the pool. It is
**defaulted off**. `docs/changelog/2026-08-07-training-item-universe.md`.

Lowering it below 100K is available but no longer buys much in absolute terms: booster fitting is
only 28.1s, so the saving would arrive mostly through CV refits. Prefer `CV_STEP_DAYS`, which cuts
the same phase directly without shrinking item coverage further.

---

## Optimization Levers

### Already applied — do not re-propose

| Lever | Applied | Effect |
|---|---|---|
| Drop p10/p90 quantiles (`QUANTILES = [0.5]`) + conformal band | 2026-08-05 | −24 models, −223.2s. `forecast_low/high` still populated, now from `conformal.band` — no frontend change was needed |
| Collapse the ensemble 3→2→1 (`N_ENSEMBLES = 1`) | 2026-08-05 | −2/3 of remaining booster fits. Unmeasured on accuracy; `N_ENSEMBLES = 2` is the first restore step |
| Remove DART (`BOOSTING_TYPE = "gbdt"`) | 2026-08-05 | −297.1s of the pre-rewrite budget; 14d **improved** +3.14pp |
| Delete residual stacking (Ridge on LightGBM residuals) | 2026-07-25 | Removed an unbounded serving-time extrapolation; `scikit-learn` left `requirements.txt` |
| Retire the momentum fallback (`MOMENTUM_FALLBACK_HORIZONS = []`) | 2026-07-24 | The classifier beats momentum at every horizon, 30d included |
| `max_feature_rows` 700K → 100K | Jul 2026 | The 4.5× lever, in the cheap direction. See the row-budget section |
| `MAX_BIN` 255 → 63 | Jul 2026 | Roughly halves histogram build cost |
| 7d HP search reduced (`N_TRIALS_MAP[7]` 15 → 10) | 2026-07-26 | Now worth ~0s: HP is cached on warm retrains and 39.4s total when cold. A full skip is available and not worth the edit |
| `SKIP_HP_HORIZONS = [3]` | Jul 2026 | 3d frozen on its 50-trial winner |
| Regime training skipped on warm retrains | Jul 2026 | `forecaster.py:3076` — also skipped whenever `SKIP_REGIMES=1`. ⚠️ **An earlier version of this row said `meta.json` currently carries `trained_regimes: []`, i.e. that regimes are effectively not trained. That is wrong.** The model-cache restore step in `price-forecast.yml` is `if: mode == 'predict-only'`, so the Monday `mode=full` run is **always cold**, `_warm_retrain` is False, and CI never sets `SKIP_REGIMES`. The deployed artifact carries `trained_regimes: ['bull','range','bear']` and 7 regime boosters, and `predict()` uses them — **95.4s / 10.9%** of the shipped retrain (re-measured 2026-08-09; was 54.1s / 11.1% under early stopping). A local retrain produces them too when it is cold and does not pass `SKIP_REGIMES=1` — 8 boosters, `trained_regimes: ['bear','bull','range']` — but the documented local command *does* pass it, so **the served model depends on where it was trained** (2026-08-08) |
| Feature-group permutation validation skipped on warm retrains | Jul 2026 | `forecaster.py:3296`; also auto-skipped when the val window has <2000 rows or <7 distinct dates, where the permutation test is pure noise and caused false-positive pruning that collapsed 14d/30d to ~4 features |
| Voted frame cache in `fetch_price_history` | 2026-07-29 | **35s** measured, against a ~10 min estimate in a since-deleted planning doc — the estimate was wrong by ~17×. `VOTED_CACHE=0` disables. **Bump `VOTED_CACHE_VERSION` when voting or the DuckDB query changes**, or a stale frame silently trains the next model |
| Engineered feature cache on the predict path (3-day TTL) | Jul 2026 | Removes feature engineering from most predict runs |
| Predict tail truncation (`PREDICT_TAIL_ITEM_DAYS = 240`) | 2026-08-04 | 3.59M → 1.39M rows in 1.4s |
| Drift-triggered retrain removed (report-only) | 2026-08-04 | −465s of an 835s daily step. `ALLOW_DRIFT_RETRAIN=1` restores it |
| Row sampling: `bagging` for all quantiles | 2026-07-29 | GOSS reverted after A/B showed +5.21% (3d) / +1.76% (7d) pinball and +1.13pp / +0.71pp DA for bagging |
| Dead-item filter, ±500% target winsorization, corrupt-item flagging | 2026-07-17 | Removed ~41% of training rows that carried no signal |
| Allowlist before the correlation prune **+** skip the discarded feature blocks | 2026-08-09 | **−16.5s combined, measured paired on the production frame** (24.5s → 8.0s, warm voted cache, both arms same code). ⚠️ The planning doc sized these separately at 25.2s + 8.5s = 33.7s; that did **not** reproduce, so quote 16.5s. `df[feature_cols].corr()` is O(rows × p²) single-threaded pandas and was running over 123 candidate columns instead of the 33 the allowlist keeps. **Verified output-preserving**: flipping `ALLOWLIST_BEFORE_PRUNE` and `skip_unused_groups` yields set-identical 33 features, matching the shipped `meta.json`. Not identical *by construction* though — `_prune_features` keeps the lower-indexed member of a >0.95 pair and index order does not follow group, so `ALLOWLIST_BEFORE_PRUNE = False` exists to restore the old order. The skip half is default **off**: seven `ab_test_*` harnesses build their own frame and need the full 123 columns, so only `build_training_data` passes `skip_unused_groups=True`. Its skip set is *derived* from the allowlist, never a literal — re-admitting `cross_sectional` (Track C4) must not yield a frame with the group allowlisted and its columns absent, median-filled to zero |
| Cap CV fold training rows (`CV_MAX_TRAIN_ROWS = 300_000`) | 2026-08-09 | `max_rows` was applied only in `_build_production_split`, so `_cv_evaluate_horizon` took the whole expanding window every fold — nine folds per horizon summed to **4.2× the training frame**. Only `train` is thinned; fold count, val rows and OOF record count are unchanged, because they are the sample size of `q_hat`, `mean_rank_ic` and the PT statistic. The fold model now fits on less data than the served one, so `q_hat` comes out **larger** — over-coverage, the safe direction. **Verify empirical coverage against `NOMINAL_COVERAGE = 0.80` before lowering it** |
| Optuna scores within-date rank IC | 2026-08-09 | **A correctness fix that COSTS time — the largest single phase of the cold retrain is now HP search.** The objective was still `best_score["valid_0"]["quantile"]` under `lgb.early_stopping(20)`, i.e. the criterion `FIXED_BOOST_ROUNDS` replaced on 2026-08-08, with a pruner cutting trials on LightGBM's reported metric rather than on the returned objective. At 14d/30d the val-loss optimum is 25 rounds while rank IC peaks at 500–750. Removing early stopping means every trial now trains the full `_boost_rounds(horizon, cv=True)` instead of stopping at ~25 rounds: measured **392.3s of a 1426.3s** cold retrain (0.0 / 107.8 / 37.6 / **246.9s** at 3/7/14/30d, 3d frozen by `SKIP_HP_HORIZONS`). Budget for it, or cut `N_TRIALS_MAP[30]`. `MODEL_ARTIFACT_VERSION` 5 → 6 |
| Content-hashed archive fingerprint | 2026-08-09 | The voted-frame cache keyed on `st_mtime_ns`, and CI checks the archive out fresh every run, so the key changed unconditionally and **the cache could never hit in CI** — ~48s a run (21.1s DuckDB read + 27.2s voting), paid by the daily predict path too. Now row count + byte size, naming no column: the archive is not schema-uniform and its date column is `day`, not `date` |

### Still available

| # | Lever | Change | Speed gain | Quality risk |
|---|-------|--------|-----------|--------------|
| **1** | **Widen the CV stride** | `CV_STEP_DAYS` 150 → higher. Env-overridable, no code edit (`forecaster.py:316`) | Cuts folds ~linearly against a **439.3s** phase (33 folds, 50.4% of the retrain) | **Real and structural, not statistical.** Folds are the conformal calibration set *and* the confidence-threshold fit set, so fewer folds means fewer OOF points, a noisier `q_hat`, and a looser coverage guarantee. Directional accuracy is also clustered by date, so folds are the effective sample size of every CV number in this doc. ⚠️ **Fold count is now also the input to the offline rank-IC / Pesaran–Timmermann metrics, not just conformal calibration** — cutting folds degrades the evaluation as well as the band. Verify empirical coverage against `NOMINAL_COVERAGE = 0.80` before and after |
| **2** | ~~Stop computing discarded features~~ — **APPLIED 2026-08-09**, see the row in the table above | — | — | — |
| **3–6** | ~~`MAX_BIN` 63→31, `num_leaves` 47→31, aggressive correlation pruning, more regularization~~ — **REFUTED 2026-08-09, do not re-propose.** Measured per boosting round on the production frame: `max_bin` 31 → **26.3 ms**, `num_leaves` 31 → **26.2 ms**, `min_data_in_leaf` 100 → **27.1 ms**, `feature_fraction` 0.4 → **28.9 ms**, against a **25.5 ms** baseline. Every one is *slower* than doing nothing. The "~10–20% of 28.1s" figures in the earlier version of this table were estimates, never measurements | — | — |
| **7** | Drop the 14d horizon | `HORIZONS = [3, 7, 30]` | ~25% of train + inference; −2 models | **Unjustifiable in either direction right now.** The old rationale ("loses a 55.7% DA horizon") came from the superseded scorer and there is no quotable production DA to replace it with. 14d also has the best classifier CV DA of the four and gained the most from removing DART (+3.14pp), so it is the worst horizon to cut, not the safest. Do not pull this without a measurement |
| **8** | LightGBM → ONNX for inference | Convert the boosters | 0 on training, and applies only to the ~15s of booster scoring — not the 72s of fetch + feature engineering | ~0pp if the conversion is exact. Low ceiling: fix the data path first |
| **9** | Lower `TRAIN_FEATURE_ROWS` below 100K | Env var | Mostly through CV refits | Cuts item coverage below 1.8%. Prefer lever 1 |

### 🛑 Do not

| Lever | Why |
|---|---|
| **Parallel ensemble or horizon training** | Removed deliberately. Both the horizon `spawn` Pool and the ensemble `ThreadPoolExecutor` were deleted on 2026-07-21 (−209 lines) because they deadlocked under OpenMP and the surrounding timeouts were masking it. `AGENTS.md` documents training as fully sequential; LightGBM's OpenMP threads supply the CPU parallelism. Re-adding this reintroduces a fixed bug. See `docs/changelog/2026-07-21-remove-training-parallelism.md` |
| **`SKIP_CV=1` in CI** | `q_hat` is derived from CV out-of-fold predictions. Skipping CV routes calibration to a single holdout that is *also* the early-stopping and Optuna scoring set, so the band under-covers. It was removed from `price-forecast.yml` and is pinned by `test_ci_workflow_does_not_skip_cv`; the rationale is repeated at `forecaster.py:3141-3160`. It survives as a local/dispatch speedup only |
| **CatBoost** | Tested Jul 2026, degraded accuracy 18–20pp. No longer a dependency |
| **Neural forecasters** (N-BEATS / PatchTST / TFT) | Slower, GPU-dependent, unknown on this data — and not worth attempting while the gate cannot resolve 1pp |
| **Cap GBDT rounds at 500** | Moot. The lever existed because q10 was still improving at ~774 rounds; q10 no longer exists, and the surviving q50 models stop well under 500 |
| **Drop the 5 social features as a speed lever** | Moot as stated. The allowlist already discards them and the collector's workflow is deleted, so they are identically zero. Only the feature-engineering cost remains, which is lever 2 |

---

## Recommended order

1. **Lever 1 (CV stride), with a coverage check.** Retrain and compare empirical band coverage
   against the 80% nominal before and after. It acts on the 439.3s / 50.4% phase, and its risk is
   structural rather than statistical — note that folds now also carry the offline rank-IC/PT
   metrics.
2. ~~**Levers 3/4/5/6**~~ — **measured dead 2026-08-09.** All four are slower per boosting round
   than the baseline. Do not bundle them into a retrain; do not re-propose them.
3. **Nothing on the predict path until the fetch is addressed.** 72 of 87s is fetch + feature
   engineering; lever 8 optimizes the remainder.
4. ~~**Lever 2**~~ — **applied 2026-08-09**, and it was worth ≈**32s**, not the "~1% of the
   retrain" this doc claimed. The re-size came from measuring the two halves separately: **25.2s**
   was `_prune_features` building a correlation matrix over 123 columns instead of 33 — which this
   doc never mentioned at all, and which is the larger half — and **8.5s** was the discarded
   feature blocks themselves. Both are in the applied table above. The lesson is the one this
   section keeps relearning: the sizes here were estimates, and three of them have now been wrong
   by more than an order of magnitude in both directions.

The two levers not in this table but larger than any of them are in
`docs/research/2026-08-08-model-review.md`: gating the per-fold diagnostic classifier off in CI
(**already done** — measured −932s, −52%, touches nothing served) and deciding whether the
**95.4s** of regime models earn their place. Both re-measured 2026-08-09; the first is spent, so
the second is the only large lever left.

---

## Verification Protocol

After applying any change:

```bash
cd backend
venv/bin/python -m pytest tests/test_forecaster.py tests/test_minimal_model_shape.py -x -q
SKIP_REGIMES=1 FORCE_HP_SEARCH=1 venv/bin/python scripts/forecast_prices.py --train-only
venv/bin/python scripts/walkforward_backtest.py       # fresh-model DA gate
```

Run the whole suite as **`pytest tests`**, never bare `pytest` — `scripts/test_social_signal.py`
imports `thefuzz`, which is not in `requirements.txt`, and collection aborts with
"714 tests collected, 1 error … Interrupted".

**Do not use `SKIP_CV=1` when anything touching the median model or the band changes.** Conformal
`q̂` and the confidence thresholds are both fitted on the CV out-of-fold predictions that populate
`ItemForecaster.conformal_calibration`, so skipping CV leaves a stale calibration behind a changed
model.

**`--predict-only` no longer retrains** (fixed 2026-08-04). It used to drift-check every horizon
and auto-retrain via `forecaster.train(max_rows=700_000)` — with no `SKIP_REGIMES` — whenever a
horizon fell below a 60% DA threshold. Because that threshold sits above anything the model has
ever measured, it fired on **every** run: 465s of an 835s daily step, with the served forecasts
coming from a throwaway warm retrain rather than the scheduled model. Drift is now reported only,
and `check_concept_drift` additionally ignores accuracy rows that lack forecast-date coverage. Set
`ALLOW_DRIFT_RETRAIN=1`, or dispatch with `mode=full`, to retrain — noting that `mode=full` still
only trains if the artifact is ≥14 days old or `FORCE_RETRAIN=1`.

**Which accuracy script to use:**

| Script | Measures | Reflects a fresh model? |
|---|---|---|
| `scripts/walkforward_backtest.py` | Retrospective walk-forward folds using current tuned params | **Yes** — the fresh-model gate. But it bypasses `fetch_price_history`, so it scores a plain-mean price consensus over a different item universe than production trains on |
| `scripts/backtest_accuracy.py` | Stored forecasts vs matured actuals | **No** — 3–30 days for maturity, further bounded by archive coverage. This is what CI runs |
| `scripts/compute_mde.py` | Minimum detectable effect for a proposed A/B | Run this **first** |

### There are no retention floors

This section used to carry a "retain ≥90% of current accuracy" gate with per-horizon floors
(3d ≥55.4%, 7d ≥47.5%, 14d ≥50.1%, 30d ≥48.8%). Those floors were derived from pre-2026-08-02
production DA, which the deterministic-backtest fix showed to be an artifact of a scorer whose two
legs used different estimators — so a naive reading had all four horizons failing a gate they were
never actually measured against.

There is nothing to replace them with. `MIN_FORECAST_DATES = 20` means no production DA is
publishable yet, and CV DA, walkforward DA and production DA each score a different item universe
and a different price consensus, so none can stand in for another.

Until a post-rewrite cohort matures past 20 forecast dates, judge an optimization on a
**pre-registered paired comparison between two arms of the same harness**, with the MDE computed in
advance — the method used for the minimal-model rewrite
(`docs/changelog/2026-08-04-minimal-model-results.md`). Do not accept or reject a change against a
floor derived from a different scorer.
