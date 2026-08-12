# Model Architecture

> **⚠ There is no reportable production directional-accuracy number right now.**
> `MIN_FORECAST_DATES = 20` (`backtest/scoring.py`) refuses to publish a headline below 20
> distinct forecast dates, and every live cohort currently spans **1–2** dates — so all four
> horizons report NO HEADLINE. That is a calendar problem, not a code problem: the cohorts have to
> accumulate dates. The CV figures below are training-time diagnostics measured on a different
> item universe and a different price consensus, and they are **not** comparable to a production
> number. See `docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md`.
>
> **And when it arrives it will not be a DA.** The published headline is a Pesaran–Timmermann
> test — see "The headline is a test, not an accuracy" below. Raw DA is never quoted alone.
>
> Two corrections to the paragraph above. The panel is **8 / 7 / 4 / 1** dates at 3/7/14/30d since
> `served_identity()` stopped keying the cohort on the config
> (`changelog/2026-08-11-model-version-is-not-a-config.md`), not 1–2, and half of the failure was a
> code problem. And 20 dates will not make `DA − realised_down_rate` quotable: its per-date sd is
> **15–24pp**, so pooled excess needs ~50 dates
> (`changelog/2026-08-11-the-da-gap-is-the-market-direction-of-five-dates.md`).

## Overview

`ItemForecaster` (`backend/models/forecaster.py`) trains and serves **8 LightGBM models** —
one median regressor and one directional classifier per horizon:

```
3d horizon:  1 × q50 GBM  +  1 × 3-class directional classifier
7d horizon:  1 × q50 GBM  +  1 × 3-class directional classifier
14d horizon: 1 × q50 GBM  +  1 × 3-class directional classifier
30d horizon: 1 × q50 GBM  +  1 × 3-class directional classifier
```

`HORIZONS = [3, 7, 14, 30]` (:150), `QUANTILES = [0.5]` (:156), `N_ENSEMBLES = 1` with
`ENSEMBLE_SEEDS = [42]` and `ENSEMBLE_FEATURE_FRACTIONS = [0.7]` (:193-195). There is no
ensemble averaging and no p10/p90 model. The 40-model grid this replaced (36 global + regime
files) collapsed on 2026-08-05; see `docs/changelog/2026-08-04-minimal-model-results.md`.

**Division of labour:** the classifier supplies the served **direction + confidence**; the q50
model supplies the **median**; conformal calibration supplies the **band**. Nothing else
contributes to a served forecast.

**Model version:** `lgbm-v3` (`scripts/forecast_prices.py:35`)
**Artifacts** in `backend/models/saved_models/`: `lgb_{horizon}d_q50_e0.txt` (:4897),
`clf_{horizon}d.txt` (:4921), `bias_corrections.json`, `meta.json`.
`MODEL_ARTIFACT_VERSION = 5` (:174) — `load_models()` raises `IncompatibleModelArtifact` on an
older cache rather than serving a band computed by a different scheme. v4 (2026-08-06) marks
the as-of lag lookup, which changes feature *values* and the persisted `feature_medians`
without changing the set; v5 (2026-08-06) marks the dollar-scale columns leaving the feature
set, after which a v4 booster's splits are thresholds in dollars and are meaningless.

### Directional classifier

A per-horizon 3-class (down/flat/up) LightGBM trained on multiclass log-loss, which optimizes
the served metric directly. Labels come from a fixed ±`DIRECTION_FLAT_TOLERANCE_PCT = 0.5`
band on the realised return (:28). Movers (`|return| > tolerance`) are up-weighted 3.0×
(`DIRECTION_MOVER_WEIGHT_MAP`, :276) so the classifier spends capacity on the hard calls rather
than the large easily-predicted flat mass. Predicted class = `argmax`; confidence is `high` when
`max(prob) >= DIRECTION_CONFIDENCE_HIGH = 0.5` (:284, applied :4296-4300).

**Vol-scaled labels are dead code.** `DIRECTION_VOL_MULTIPLIER_MAP` (:277), the `label_vol_30d`
column, and `_direction_threshold` all exist, but every call site passes
`sigma_train=None, sigma_val=None` (:3519), so the threshold is always the fixed scalar. The
branch is unreachable in training and in CV.

### Conformal prediction band

`models/conformal.py` (106 lines) replaces the 24 p10/p90 quantile GBMs, which cost 223.2s of a
381.2s training budget for 39–48% empirical coverage against an 80% target.

| Step | Where | What |
|------|-------|------|
| Per-item scale | `conformal.sigma_from_columns` (`conformal.py:48`) | `sigma = price_std_60d / price` — the 60-day coefficient of variation, in return space |
| Clip bounds | `conformal.sigma_bounds` (`conformal.py:31`), called `forecaster.py:2820` | 1st/99th percentile of the training frame's sigma distribution, frozen into `meta.json` |
| Calibration | `conformal.calibrate` (`conformal.py:72`), called `forecaster.py:3202` | `q_hat` = finite-sample-corrected quantile of `|residual| / sigma` over pooled out-of-fold CV residuals. Dimensionless |
| Serving | `conformal.band` (`conformal.py:98`), called `forecaster.py:4219` | `mid ± q_hat * sigma`, symmetric in return space |

`NOMINAL_COVERAGE = 0.80` (`conformal.py:23`). Because the band is symmetric about the median it
**cannot cross by construction**, which is why `predict()` no longer repairs quantile ordering.
`_fix_quantile_crossing` (:3775) survives only for `walkforward_backtest.py`'s baseline arm and
the `ab_test_*` scripts; `tests/test_minimal_model_shape.py` asserts it is absent from
`predict()`. The band is deliberately *not* floored at −100% in the band step — truncating one
side would break the symmetry the coverage guarantee rests on — so `_sanitize_forecasts` (:4348)
is what guarantees the served triple is ordered and positive.

`predict()` raises rather than serving a forecast if `q_hat` is missing for a horizon
(:4207-4212).

### Regime models

The code still supports per-regime ensembles (`REGIMES = ["bear", "range", "bull"]`, :185,
thresholds ±3% on `market_return_30d`, :186-187), which at the current grid would be at most
3 × 4 = 12 extra models. A regime is skipped below `MIN_REGIME_TRAIN = 500` train rows (:3091).
⚠️ **An earlier version of this paragraph read "Nothing trains them in production: CI passes
`SKIP_REGIMES=1`, `meta.json` carries `trained_regimes: []`." That is wrong.** `price-forecast.yml`
never sets `SKIP_REGIMES`, and the Monday `mode=full` run is always cold (the model-cache restore
step is `if: mode == 'predict-only'`), so `_warm_retrain` is False and regimes *are* trained and
served. A cold retrain produces 8 regime boosters and `trained_regimes: ['bear','bull','range']`,
at **95.4s / 10.9%** of the retrain (measured 2026-08-09). `predict()` falls back to
`no regime models trained, using global` (:4131) only for artifacts built with `SKIP_REGIMES=1` —
which is what the documented *local* retrain command passes, so a locally trained model and a
CI-trained one do not serve the same thing.

---

## Features

`_feature_group()` (:115-137) partitions engineered columns into nine groups:
`price_technicals`, `supply_depth`, `item_identity`, `item_metadata`, `temporal`, `events`,
`cross_sectional`, `social`, `other`.

**`FEATURE_GROUP_ALLOWLIST = ["price_technicals"]`** (:226, applied :2529-2536). This is the
single most consequential fact about the feature set: **every other group is computed on every
training row and then discarded.** Temporal, event, cross-sectional, rarity/identity, supply-depth
and social features all cost feature-engineering time and contribute nothing to any model. An
ablation (2026-07-24, purge-gap CV) found the 85 non-price features added no measurable
directional accuracy and hurt 3d/30d, so the columns stay computed but unused rather than being
deleted — the `ab_test_*` scripts build their own feature lists from the frame and need them.

Two further filters run before the allowlist:

- **`SHELVED_FEATURES`** (:260-327) — **56** named columns withheld by name because they resolve
  to `price_technicals` and the allowlist would otherwise pass them straight through. Three
  groups:
  - **Six** volatility-asymmetry / oscillator-divergence primitives shelved 2026-07-31
    (`docs/changelog/2026-07-31-price-primitives-shelved.md`).
  - **Thirteen** volume features: the archive's `volume` column has been identically **0** since
    2026-05 — stored as 0, never NULL, which defeats every guard in the volume feature code
    (`has_volume` tests `notna()` so it stays True, `volume_missing` reports "present") — so they
    carry real signal on pre-2026-05 training rows and are dead on 100% of served rows.
  - **Thirty-seven** dollar-denominated columns, `_DOLLAR_SCALE_FEATURES` (:318-326), shelved
    2026-08-06: all `price_std_*`, `price_mean_*`, `price_min_*`, `price_max_*`, `price_lag_*`,
    plus `price_log`, the raw MACD trio and `bb_upper`/`bb_lower`. The target is a **percentage**
    return, so a dollar-scale input can only encode item identity — and on the 2026-08-06
    artifact these carried **55.6 / 70.2 / 77.5 / 86.6%** of total gain at 3/7/14/30d against a
    training median price of $0.086 and served items reaching $639. `price_tier` is deliberately
    kept: a bounded categorical is the honest way to express price level.
- **Correlation pruning** at `PRUNE_CORRELATION_THRESHOLD = 0.95` (applied after shelving).

All shelved columns are still **computed** — the conformal band reads `price_std_60d`,
`_apply_market_aggregates` reads `price_std_30d`, and the z-score / Bollinger / MA-distance /
support-resistance / log-return features are all derived from the means, mins, maxes and lags.

Net effect: the shipped 2026-08-05 artifact carried **47 feature columns**; shelving the volume
features took the 2026-08-06 artifact to **36** (`meta.json: horizon_feature_cols`, identical for
all four horizons), and shelving the dollar columns takes it to **32** on the next retrain, on a
training smoke run. The `MODEL_ARTIFACT_VERSION` bump exists to force that retrain rather than
wait 14 days for the age trigger.

`HORIZON_EXCLUDED_GROUPS` (:215-218) still excludes `cross_sectional` from 14d and
`cross_sectional` + `events` from 30d, but the allowlist already removes both from every horizon,
so it is currently a no-op.

### Price technicals (the only group that reaches a model)

Lags at `LAGS = [1, 3, 7, 14, 30, 60, 90, 120, 180]` and the returns/log-returns derived
from them; rolling mean/std/min/max over 7/14/20/30/60d and the coefficient-of-variation forms
`price_cv_{7,14,20,30,60}d`; distance from the 100d and 200d moving averages; Bollinger Bands
(20d: upper, lower, %B, width); RSI(14); MACD line/signal/histogram plus the price-normalised
`macd_line_rel` / `macd_histogram_rel`; support/resistance distances; high/low range; price
acceleration; autocorrelation proxies; trend divergence.

**Everything that reaches a booster is scale-free** (2026-08-06), `price_tier` excepted. The raw
dollar levels are computed and then shelved; the model sees their return-space forms.
`price_cv_60d` is by definition the same quantity as the conformal band's
`sigma = price_std_60d / price`, and `tests/test_scale_free_features.py` pins both that identity
and the general property — it multiplies every price by 100 and asserts no served feature moves,
so a dollar-scale column added later fails without anyone updating a name list.

Lag and return features are looked up **by calendar date**, the same way targets are, so an
archive day gap yields NaN → median-fill rather than a fabricated multi-month return. The lookup
is *as-of* within `LAG_TOLERANCE_DAYS = 3` (:179): the aggregator drops whole calendar days
(August 2026 held only 08-01 and 08-04) and an exact-date lookup NaN'd that lag for every item at
once. Holes wider than the tolerance still yield NaN — see Known limitations.

### Computed and discarded

Volume, temporal (day/month/quarter + sin/cos encodings, item age), event decay + density,
cross-sectional market return/volatility/regime flags, rarity ordinal and one-hot (source:
`price-archive/item-metadata.parquet`), supply depth, and 5 social sentiment columns. The social
collector scores with **FinBERT ONNX INT8** (`collectors/social_sentiment.py`), not VADER, but
`reddit-sentiment.yml` was deleted after runner IPs started getting 403s, so the source table
holds 0 rows and the features are identically zero regardless.

---

## Training Pipeline

`train()` (:2784) runs one horizon at a time via `_train_horizon_inline()` (:2844, called :2833).

### Training row budget

The most consequential knob in the system. Two separate budgets, previously conflated:

| Budget | Default | Where | What it bounds |
|--------|---------|-------|----------------|
| `max_feature_rows` | 100,000 | `DEFAULT_TRAIN_FEATURE_ROWS`, `forecast_prices.py:45`; env override `TRAIN_FEATURE_ROWS` (:54) | The frame **before** feature engineering — i.e. how many whole item histories the model ever sees |
| `max_rows` | 700,000 | `TRAIN_HORIZON_MAX_ROWS`, `forecast_prices.py:50` | Each horizon's slice **after** feature engineering |
| `min_median_price` | `None` | `_train_min_median_price()`, env `TRAIN_MIN_MEDIAN_PRICE` | Which items the budget may buy: a floor on each item's median price, applied **before** the subsample |

`_stratified_item_subsample()` (:2419, applied :2496-2499) spends the first budget by selecting
**entire item histories** (stratified by rarity, full calendar window preserved) so lags and
rolling features stay valid. It is therefore an *item-coverage* budget, and at the default it
selects **99 of 5,377 items — 1.8% of the pool**.

**The subsample's seed is a first-order term.** It is hard-coded (`seed: int = 42`), and eight
retrains varying only that seed moved `mean_classifier_acc_ge1` with sd **1.82 / 3.05 / 1.54 /
2.77pp** at 3/7/14/30d, over near-disjoint draws (Jaccard 0.010–0.026 vs seed 42). Absolute
levels from a single retrain are therefore not comparable across runs; paired arms at a fixed
seed are unaffected. `docs/changelog/2026-08-07-training-item-universe.md`.

`min_median_price` exists because the pool is **82.56% sub-$1 item-days** while
`MIN_SERVED_PRICE_USD = 1.0`, so at the default budget only ~20 of the 99 selected items are in
the served cohort. The ≥$1 cohort is **926 items / 993,464 item-days** at essentially the pool's
own density, so a floor of 1.0 with `TRAIN_FEATURE_ROWS` ≥ 1.0M trains on the whole served cohort
with **no subsample** — 538s, and the in-model permutation test goes from 7 WARN in 32
horizon-runs to 4/4 PASS. It has been **production's default since 2026-08-08** (floor 1.0 +
1.2M budget, `DEFAULT_TRAIN_MIN_MEDIAN_PRICE` / `DEFAULT_TRAIN_FEATURE_ROWS`), and it shipped
**on determinism, not accuracy**: removing the draw removes the seed's 1.5–3.1pp variance. The
accuracy evidence is withdrawn — the stored +3.50pp at 30d re-derives to **+1.642pp
[−0.809, +4.505], null**
(`docs/changelog/2026-08-08-per-fold-price-filter-rederived.md`). See the changelog also for
the caveat that `predict()` still writes sub-$1 rows.

Raising it was measured and declined: 700,000 rows selects 646 items and costs **468.7s** against
**104.6s** at 100,000 (:2801-2806). That is more than the pre-rewrite 40-model grid cost — the
entire minimal-model saving spent to buy 12% of the item pool. The fresh-model gate cannot
resolve the accuracy difference either, so there is no measurement that would justify it. The two
budgets are deliberately not unified: one number would otherwise move coverage and the
per-horizon cap at the same time.

### Training window

`days_back=1460` (:2807), backfilled items only, read from `price-archive/*.parquet` via DuckDB.
The 2026 distribution-shift guard that used to exclude the current year was removed once the
May–June 2026 archive gap was backfilled (:2487-2494).

### Item universe

Three exclusions are applied at the read, not downstream, so training and `predict()` see the
same universe from the one query in `_fetch_voted_price_history`:

| Rule | Where | What it removes |
|---|---|---|
| `BID_SOURCES` | `models/item_parser.py`, dropped in `_apply_multi_source_voting` | `aggregator_buff163_buy` — a bid, which must not vote against asks |
| `TRAILING_WINDOW_SOURCES` | `models/item_parser.py`, dropped in `_apply_multi_source_voting` alongside `BID_SOURCES` (2026-08-09) | `aggregator_steam_7d/30d/90d` — Steam trailing-window MEAN sale prices, the wrong time basis rather than the wrong side of the book |
| `PHASE_COLLAPSED_SLUG_PATTERNS` | `models/item_parser.py`, applied as `phase_collapsed_sql_filter()` | Doppler / Gamma Doppler names, whose returns are phase-composition artifacts |

A `market_hash_name` encodes weapon + finish + wear + StatTrak/Souvenir and nothing else, so a
Doppler name is **29 base names covering 181 distinct `paint_index` assets**; the quoted
headline is the *cheapest* phase **95.5%** of the time, so the series steps whenever which phase
is cheapest changes — a level shift with no asset repricing. The rule lives in `item_parser.py`
(light enough for `api/` to import without LightGBM) and is re-exported from `forecaster.py`; it
is applied at `_fetch_voted_price_history`, both `walkforward_backtest` loaders and
`api/routes/opportunities.py::_load_items`. `PHASE_COLLAPSED_EXEMPT_PATTERNS = ("sticker",)`
keeps the two measured false positives (`Sticker | Doppler Poison Frog (Foil)` and its Sticker
Slab twin). `docs/changelog/2026-08-08-phase-collapsed-names-dropped.md`.

### Label hygiene

`prepare_targets` winsorizes `target_return_{h}d` at ±500%, and then (2026-08-06) voids labels
built across days the collector fabricated. Winsorization cannot catch these — the returns are
well inside the clip and survive as confident, wrong labels.

| Defect | Detector | Threshold | Rule | Incidence |
|---|---|---|---|---|
| Re-published snapshot (a day that is a byte copy of the previous one) | `_snapshot_dates` | `SNAPSHOT_DAY_FLAT_FRACTION = 0.99`, `MIN_DEGENERATE_CROSS_SECTION = 25` | Bad **endpoint** only — a copied day shifts no level, so it is harmless mid-window | 2 of 4,735 archive days (2026-07-16, 2026-07-22), both at 100.00%; next-highest day 69.01% |
| Collector cutover (a source-regime change in the stitched archive) | `_collection_shift_dates` | `COLLECTION_SHIFT_FRACTION = 0.20` on the **item universe size** | Corrupts any label whose window **spans** it, so the whole horizon-wide anchor band is voided | 12 of 4,735 days (0.25%) — 4 in 2013, 1 in 2016, 7 in 2026 |

Cutovers are detected from the universe size and never from prices, deliberately: prices moving
cannot change how many items a collector returns, so the detector cannot mask a real crash
(`tests/test_degenerate_label_dates.py::test_a_price_crash_is_never_flagged`). The market-return
signature it catches is large — −31.6% on 2026-03-22, +17.4%/−17.8% on 2026-07-09/10, against
±0.5% on a normal day — and is a basis change, not a price move.

### Hyperparameter search

Optuna TPE with MedianPruner, per-quantile. `N_TRIALS_MAP = {3: 50, 7: 10, 14: 15, 30: 15}`
(:206) and `SKIP_HP_HORIZONS = [3]` (:210) — 3d is frozen on its 50-trial winner, warm-started
in `_optuna_search_params`. 14d and 30d still search because they are the noisiest horizons;
the original reason (tuning DART's dropout params) went away with DART. A warm retrain reuses
cached params from `meta.json` and logs `optuna: 0.0s`.

`study.optimize()` is called with no `timeout=` and there is no timeout constant anywhere in
`backend/models/`. The only wall-clock protection is `timeout-minutes: 180` in
`.github/workflows/price-forecast.yml`.

### Validation and calibration

The production train/val split in `_train_horizon_inline` is **purged**
(`_purge_overlapping_train_rows`, 2026-08-06): a row dated `d` is labelled with the price at
`d + horizon`, so rows in `[split_date - horizon, split_date)` carry labels drawn from the
validation window — at `horizon == VALIDATION_WINDOW_DAYS == 30`, the entire window. That frame
is the `dval` early stopping stops on, the set Optuna scores every trial against, and the
classifier's stopping set, so before this every shipped tree count, hyperparameter and stopping
point was selected against partly-seen labels. The positional fallback split is purged the same
way. CV always purged.

Expanding-window CV: `CV_STEP_DAYS = 150` (:306), `VALIDATION_WINDOW_DAYS = 30` (:201),
`CV_MIN_TRAIN_DAYS = 200` (:307), each fold carrying a `horizon`-day purge gap. A 1460-day frame
yields 8–9 folds. Folds report persistence and momentum baselines and `edge_vs_best_baseline`,
judged on the **classifier** accuracy because that is the served signal (:3240-3246); the ≥$1
cohort accuracy is reported alongside the all-tiers number so it is comparable to the production
headline (:3248-3255).

CV is the calibration set, not just a diagnostic: `q_hat` and the confidence thresholds are both
fitted on pooled out-of-fold residuals (:3186-3222), in that order. This makes it the dominant
cost — **~148.6s of a 176.7s warm retrain, ~84%** — because it refits a median model per fold per
horizon purely to generate residuals.

**`SKIP_CV=1` is deliberately not set in CI** (`price-forecast.yml`, pinned by
`test_ci_workflow_does_not_skip_cv`; rationale at :3141-3160). It routes to a single-holdout
fallback that fits `q_hat` on the same rows used for early stopping and Optuna scoring, so the
band under-covers; `train()` logs that at WARNING (:3214-3221). It survives as a local/dispatch
speedup only.

### Boosting

`BOOSTING_TYPE = "gbdt"` (:205) for all four horizons. There is no per-horizon boosting map and
no DART: when DART was finally measured against GBDT on a trustworthy gate, 14d **improved**
+3.14pp (CI [+1.955, +4.373]) and 30d was unchanged — and 14d was the horizon DART was supposedly
earning its cost on. The dropout branches went with it.

`data_sample_strategy` is `bagging` for the median model (:2061); GOSS was reverted 2026-07-29
after it measured worse under the quantile objective's constant ±alpha gradients.

### Parallelism

**None.** Horizons, quantiles and ensemble members train sequentially; LightGBM's OpenMP threads
supply the CPU parallelism, with ensemble members getting `n_jobs = cpu_count // 2` and the
Optuna search params keeping `n_jobs: -1`. Both the horizon `spawn` Pool and the ensemble
`ThreadPoolExecutor` were deleted on 2026-07-21 (−209 lines) because they deadlocked under
OpenMP and the surrounding timeouts were masking it. See
`docs/changelog/2026-07-21-remove-training-parallelism.md`. **Do not re-add this.**

`_gpu_available()` (:84) still probes CUDA in a subprocess — required because
`lgb.train(device="cuda")` segfaults uncatchably on a CPU-only pip wheel — and sets
`device: cuda|cpu` at :2949 and :2959. Every shipped model in `meta.json` carries `device: cpu`.

### Deleted layers

- **Residual stacking (Ridge on LightGBM residuals)** — deleted 2026-07-25. It was fit on raw
  unscaled feature values, so it extrapolated without bound: a penny item with a legitimate +900%
  `return_Nd` got a correction in the +100,000% range, over-correcting 99% of 14d items and
  inverting quantile ordering on 100% of predictions. Tombstone comment at :285-293. No `.pkl`
  artifacts exist and `scikit-learn` is not in `requirements.txt`.
- **CatBoost** — removed; not a dependency. The stale `backend/catboost_info/` directory it left
  behind was deleted 2026-08-10.
- **Momentum fallback** — `MOMENTUM_FALLBACK_HORIZONS = []` (:266), superseded by the classifier,
  which beats momentum at every horizon including 30d. `_recenter_on_momentum` (:3445) survives
  as an unreachable path behind that empty list.

### Retrain triggering

**Age-based only.** `RETRAIN_INTERVAL_DAYS` defaults to **14** (`forecast_prices.py:208`) and a
retrain fires when the artifact's age reaches it, or when `FORCE_RETRAIN=1` (:207).

Drift is **report-only**. `DRIFT_DA_THRESHOLD = 60.0` (:175) is now an alert threshold: the old
drift-triggered retrain compared measured DA against a 60% floor the model has never reached, so
it fired on *every* run and added a measured 465s to an 835s daily step, serving forecasts from a
throwaway warm retrain instead of the scheduled model. Set `ALLOW_DRIFT_RETRAIN=1`
(`forecast_prices.py:241`) to restore the old behaviour.

Monday sets `mode=full` in `price-forecast.yml`, but `full` **only trains if the model is ≥14 days
old or `FORCE_RETRAIN=1`**. A fresh model plus `mode=full` will not retrain — Monday is not a
guaranteed retrain.

### Measured training time

10-core Mac, `SKIP_REGIMES=1 --train-only`, 2026-08-05: **cold 250.1s** (Optuna 39.4s),
**warm 176.7s**. The pre-collapse warm figure on the same machine and flags was 381.2s. Booster
fitting fell 330.8s → 28.1s; conformal CV rose to ~148.6s, so calibration, not model fitting, is
now the bottleneck. Per-lever detail in `docs/architecture/model-optimization.md`.

---

## Prediction

`predict()` processes items in chunks of `PREDICT_CHUNK_ITEMS` (default 1000, :3958) and reuses a
3-day-TTL engineered feature cache.

### Eligibility

`PREDICT_MIN_HISTORY_DAYS = 14` distinct days in the Parquet archive (:170, applied :4033-4037) —
looser than the training floor of `MIN_HISTORY_DAYS = 30` (:166) because the live aggregator
series is still young. Restricted to `is_backfilled` items, which is **re-derived from the
archive on every run** (items with pre-2026 rows) rather than being a static flag: **5,542 items**.

### Serving transform

Features come from the last row per item, reindexed with `fill_value=0` for absent columns and
then median-filled from `meta.json: feature_medians` (:4112-4115).

### Spike smoothing

The base price used to convert percentage returns into dollars is a span-bounded 3-observation
median near the anchor (`_smoothed_anchor_prices`, :4075-4100), matching the backtest's resolver
rather than "the last three rows on file". Items whose latest price deviates >10% from that
median are logged.

### Serving order

Order matters and is asserted by tests:

1. q50 predicts the median return; the horizon is skipped unless `0.5 in preds` (:4186).
2. `conformal.band(mid, sigma, q_hat)` builds the interval (:4219). `sigma_arr` is computed once
   outside the horizon loop because it does not vary by horizon (`_sigma_for_rows`, :3757,
   called :4154).
3. The classifier predicts class probabilities (:4238-4242).
4. `_blend_returns_with_prior` blends with the previous day's forecast at
   `FORECAST_BLEND_WEIGHT = 0.15` (:296, applied :4253-4255) to damp direction flip-flopping.
5. Per-tier bias correction — threshold-based, with the additive correction only as a fallback
   when no threshold data exists (:4257-4275). `BIAS_FIT_SCHEMA_VERSION = 2` (:282) discards
   thresholds fitted without a date-coverage guard.
6. `_recenter_on_direction` moves the median onto the classifier's call **last**, preserving
   interval half-widths, so the served price is coherent with the served direction (:4280-4282).
7. `_sanitize_forecasts` (:4348) clamps NaN/INF/negative prices to `current_price` with `flat`
   direction and `low` confidence, and downgrades high confidence on zero-volume items.

> ⚠️ **The no-classifier fallback branch is a live hazard, measured 2026-08-11.** When
> `self.direction_models` has no entry for a horizon, `predict()`'s `else` branch (:6965-6976 as of
> `67324fb`) derives direction from `mid_ret` against `t_down` / `t_up`, **defaulting to
> ±`DIRECTION_FLAT_TOLERANCE_PCT` (±0.5%)**. Against a served mid whose `|mid_ret|` distribution is
> shrunk far harder than realised returns, that dead band swallows the majority of calls: on
> 2026-07-19 — the last date served before a classifier existed — it produced `flat` on **64.0%** of
> ≥$1 items at h=3 against a 23.1% realised flat rate, and cost ≥ +8.7pp of DA against a plain
> zero-threshold sign rule. Flat is 0.0% on every classifier-era ≥$1 date, so nothing served today
> is affected — but the branch fires **silently**. A WARNING plus a flat-call count is the open fix.
> `docs/changelog/2026-08-11-the-da-gap-is-the-market-direction-of-five-dates.md`.

### Confidence

Binary `high` / `low`, taken from the classifier's max class probability against
`DIRECTION_CONFIDENCE_HIGH = 0.5`. `_compute_confidence` (:4774) and the per-horizon
`confidence_thresholds` in `meta.json` are only reached on the no-classifier fallback path
(:4302-4314).

---

## Accuracy

### What is measurable

**Classifier CV directional accuracy** — the 2026-08-06 36-column retrain, 9 folds (8 at 30d).
`mean_classifier_acc` / `mean_classifier_acc_ge1` from `meta.json`; the CI arm trained on 101,407
feature rows against the local arm's 115,763, so the spread between the two columns is subsample
noise (`docs/changelog/2026-08-06-serving-down-skew-refuted.md`):

| Horizon | CV DA, all tiers (local / CI) | CV DA, ≥$1 (local / CI) |
|---------|:---:|:---:|
| 3d | 67.3% / 65.8% | 49.9% / 51.8% |
| 7d | 67.2% / 65.6% | 49.0% / 49.6% |
| 14d | 67.8% / 66.4% | 51.1% / 53.3% |
| 30d | 68.6% / 68.4% | 53.4% / 53.9% |

This is a **training-time diagnostic on interior rows**, not a production figure. The CV frame is
~83% sub-$1 items, so the all-tiers number reads close to a penny-item score; the ≥$1 column is
the comparable cohort (`MIN_SERVED_PRICE_USD = 1.0`), first populated by this retrain. Do not
quote either as served accuracy.

The ~17pp pooled-vs-≥$1 spread is the gap the 2026-08-06 scale-free feature change was aimed at:
the top features of this artifact were `price_std_*` and the raw MACD legs, all denominated in
dollars against a percentage target. That change is **unmeasured** —
`docs/changelog/2026-08-06-scale-free-features-and-fabricated-labels.md`.

The quantile-median sign is also scored in CV but is **not** the served signal and understates the
shipped model — the CV block caps fits at `num_boost_round=200` while production 3d q50 trains to
several hundred rounds.

### What is not measurable

No production directional accuracy. `backtest_accuracy.py` scores *stored* forecasts against
matured actuals, `MIN_FORECAST_DATES = 20` gates the headline, and live cohorts span 1–2 distinct
forecast dates — so every horizon reports NO HEADLINE. Accuracy is clustered by forecast date, so
a five-figure forecast count with two dates has an effective sample size of two; see
`docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md`. Every production DA
measured before 2026-08-02 is additionally an artifact of a two-estimator bug in the scorer
(`docs/changelog/2026-08-01-deterministic-backtest.md`) and must not be cited or compared against.

For a fresh-model number, `scripts/walkforward_backtest.py` is the gate — but it does not use
`fetch_price_history`, so it scores a different price consensus (plain mean vs production's
outlier-voted median) over a different item universe, and is not directly comparable to
production either.

### Production backtest pipeline

`backtest_accuracy.py` evaluates mature forecasts from `item_forecasts` against the Parquet
archive. **Both legs** of the realised return resolve through one shared estimator,
`backtest/price_resolution.py::resolve_anchors` — same observation window, same multi-source
voting, same anchor-staleness cap on each side. `item_forecasts.current_price` is stored on the
outcome for reference and is **never scored on**; using it as the base leg is the bug that let one
cohort score 61.76% and 33.74% on different days with no new data. Resolved actuals are frozen
(`base_price`, `actual_price`, `resolved_at`); only `--reresolve` moves them.

Maturity is bounded by archive coverage, `min(today, archive_max_day())`, not by the calendar.
`MAX_UNRESOLVABLE_PCT = 10.0` (`backtest/resolution_gate.py:87`) fails the run rather than
reporting a cohort riddled with guaranteed misses. Aggregates land in `prediction_accuracy` (per
price tier, ≥$1 headline), per-forecast outcomes in `forecast_outcomes`.

Tier 0 (<$1) is reported separately rather than folded into the headline: it is ~72% of the
evaluated universe, and at those prices one cent is a 20% move, so the direction label is
dominated by tick quantisation. `api/serving_policy.py` sets `MIN_SERVED_PRICE_USD = 1.0` equal to
`HEADLINE_MIN_TIER` so the population the product shows is the population the headline describes.

**Metrics per horizon:** MAE, RMSE, MAPE, wMAPE, MAPE by tier; the PT block (below); DA with
bootstrap CI and interval coverage; persistence-baseline DA/MAE, improvement in pp,
`skill_vs_baseline`; `conf_gap_pp`, `conf_high_interval_cov`, `conf_calibration_error`.

### The headline is a test, not an accuracy

`backtest/directional_test.py` computes a **serial-correlation-robust Pesaran–Timmermann**
statistic, and that — not DA — is what the log line, `GET /accuracy/headline` and the product
surfaces publish. Raw DA is quotable only as a **triple** with `constant_call_accuracy` (the best
single fixed call) and `realised_down_rate` beside it.

> ⚠️ **`constant_call_accuracy` is selected with hindsight — do not difference model DA against
> it.** `constant_call_baseline` counts the outcomes and returns whichever fixed call would have
> won, re-picked **per fold**; the implied direction is `up` on 4 of 8 folds at 30d, so its 68.99%
> average is unreachable by any strategy. **The runnable fixed call is always-down, whose accuracy
> *is* `realised_down_rate`** — the third term of the triple. Against it the served classifier is
> **+3.5 / +0.1 / −1.3 / +4.4pp** at 3/7/14/30d, not the −4 to −16pp that `edge_vs_constant_call*`
> reports. Keep the metric (it bounds how much of a horizon's DA is base rate); change the
> comparison. `docs/changelog/2026-08-10-constant-call-is-hindsight-picked.md`.

> ⚠️ **And `DA − realised_down_rate` is not readable on a handful of dates either (2026-08-11).**
> Decomposed per (horizon, forecast date) on the ≥$1 `lgbm-v3*` panel, **56% / 78% / 89%** of the
> apparent −13.12 / −14.35 / −25.52pp production deficit at 3/7/14d is the realised direction of
> the 5–7 dates in the panel; the within-date term is **−1.19 / +0.99 / −1.08pp**, and on the CV
> label basis it is **+2.70 / +0.17 / +0.42pp** — agreeing with the +3.5 / +0.1 / −1.3 above.
> Per-date sd of the excess is **14.96 / 22.40 / 21.56 / 24.39pp**, so the live draw is z = −1.0 to
> −1.6. **Do not quote pooled excess on fewer than ~50 forecast dates**; the per-date null the PT
> block estimates below is the quantity that survives a short panel.
> `docs/changelog/2026-08-11-the-da-gap-is-the-market-direction-of-five-dates.md`.

The reason is measured, not stylistic: an always-down call scored **29.4% on 2025-12-01 and 76.9%
on 2026-07-17** at 7d, against a model that says "down" 57–87% of the time whatever the date. A
fixed DA is therefore skill on one date and incompetence on the other, and a pooled DA over a few
dates mostly reports which dates the cohort contained. That observation *is* the PT null
(Pesaran & Timmermann 1992, *JBES* 10(4), 461–465).

- **Per forecast date**, excess hit rate `e_d = hit_d − P*_d`, with `P*_d = Σ_k P(pred=k)·P(act=k)`
  estimated from that date's own label distributions. Estimating the null per date is what removes
  the market effect. A **constant call has `e_d` identically zero** — that is the point.
- **Over dates**, a t-stat on the mean of `e_d` under a Newey–West (Bartlett) HAC long-run
  variance — Blaskowitz & Herwartz (2014, *IJF* 30(1)) — because carry-forward prices break the
  plain PT independence assumption. Bandwidth is the published `floor(4·(T/100)^(2/9))` rule, so 2
  at the 20-date floor.
- **Hurdle `|t| > 3.0`** (Harvey–Liu–Zhu 2016, *RFS* 29(1)), not 1.96, because this repo has run
  well over a dozen A/Bs against the same outcome series.
- Dates carrying fewer than `PT_MIN_ROWS_PER_DATE = 30` rows are dropped and counted
  (`pt_n_dates_dropped`): at tiny `n_d` the per-date null is degenerate.

`pt_verdict` is one of `skill` / `no_skill` / `perverse` / `insufficient_dates` / `degenerate`. A
significantly **negative** statistic is reported at warning level as a finding, not folded into
"no skill". Rows written before this landed carry no `pt_*` keys at all, and the API reports them
as `untested` — which is not the same claim as `no_skill`.

Both departures from textbook PT push the same way — the per-date null is higher than a pooled one
on a trending market, and the HAC variance is larger than the i.i.d. one under positive
autocorrelation — so the statistic is conservative by construction.

### Known limitations

- **The serving down-skew is largely refuted.** The "+13.4pp" 7d residual was a market-period
  artifact: the interior baseline spanned full history while serving sits on the latest day.
  Like-for-like inside one window it is **+4.5 / +1.7 / −1.2 / −2.5pp** at 3/7/14/30d, and the
  transform actually *lowers* the down-rate (60.6% on raw anchor rows → 53.4% as served). **Do
  not cite +13.4pp.** See `docs/changelog/2026-08-06-serving-down-skew-refuted.md`. The two
  underlying mechanisms were real and are both now addressed: volume features median-filled on
  100% of served rows (shelved 2026-08-06), and calendar-gap lag fills — on the 2026-08-04 anchor
  `price_lag_1d`, `return_1d`, `log_return_1d` and `autocorr_1d` were median-filled on 100% of
  served rows because the archive's entire August was 08-01 and 08-04 (addressed by
  `LAG_TOLERANCE_DAYS = 3`; the underlying ingestion gaps remain).
- **The model trains on the whole ≥$1 cohort** (926 items / 993,464 item-days) since
  2026-08-08, while `predict()` writes forecasts for **8,691 distinct items** on the latest
  date (`price-archive/ops/item_forecasts.parquet`) — so the served pool is still far wider
  than the trained one. Raising coverage on the *unfiltered* pool is a measured ~4.5× cost
  increase and remains declined. The floor shipped for determinism: its stored accuracy
  result, +3.50pp at 30d, re-derives to **+1.642pp [−0.809, +4.505], null**
  (`docs/changelog/2026-08-08-per-fold-price-filter-rederived.md`), so do not cite it as an
  accuracy gain. Prior write-up: `docs/changelog/2026-08-07-training-item-universe.md`.
- **Which 99 items is worth more than any feature tested.** sd 1.5–3.1pp on
  `mean_classifier_acc_ge1` from the subsample seed alone. Read § Training row budget before
  comparing absolute accuracy across two retrains.
- **The gate cannot resolve small effects.** The A/B harness has a ~1.15pp noise floor at 3d,
  wider at longer horizons, so anything below ~1pp is unmeasurable by design. Compute the MDE
  (`scripts/compute_mde.py`) before running one.
- **7d q50 early stopping is noise-determined.** The validation curve improves only 0.28% total on
  the production frame, so `early_stopping(50)` trips on noise (stopping-round sd 41 on mean 47).
  Root cause is a regime mismatch: the 30-day validation window has ~2× the return spread of the
  training set. Any 7d accuracy A/B is partly measuring this.
  `docs/changelog/2026-07-29-7d-q50-early-stop.md`.
- **`_apply_multi_source_voting()` uses `groupby().apply()`** over millions of rows and takes
  minutes. Vectorizable, but it affects only training/fetch time. The voted frame is cached —
  bump `VOTED_CACHE_VERSION` when voting or the DuckDB query changes. Now at **v6**: v2 marked
  the `BID_SOURCES` exclusion, v3 the phase-collapsed names leaving the universe, v4 the
  phantom slug keys, v5 added `n_ask_sources`, v6 excluded `TRAILING_WINDOW_SOURCES` (Steam's
  trailing-window means). See `.claude/rules/item-universe.md` for the full history and the
  cache-bump trigger.
- **The `ab_test_*` harnesses do not share the production universe.** Ten-plus of them carry
  private archive globs and filter neither `BID_SOURCES` nor the phase-collapsed names, so they
  train on a universe production no longer has. Tracked as step 5 of
  `docs/research/2026-08-07-next-steps.md`, alongside their missing purge and embargo.
- **Dead code that survives deliberately:** `_fix_quantile_crossing` (backtest baseline arm),
  `_recenter_on_momentum` (empty `MOMENTUM_FALLBACK_HORIZONS`), the vol-scaled direction-label
  branch (all call sites pass `sigma=None`), regime training (`SKIP_REGIMES=1` in CI), and every
  non-`price_technicals` feature group (computed, then dropped by the allowlist).

---

## Horizon Selection

| Horizon | Status | Rationale |
|---------|--------|-----------|
| 1d | ❌ Rejected | Too noisy — day-to-day CS2 price action dominated by random walk |
| 3d | ✅ Active | Short-term momentum, smooths weekend gaps |
| 7d | ✅ Active | Primary horizon |
| 14d | ✅ Active | Natural midpoint, many CS2 cycles run ~2 weeks |
| 30d | ✅ Active | Longest horizon, benefits most from the 1460d training window |
| 60d | ❌ Not yet | Would require more data and richer long-term features |

---

## What NOT To Do

- **Do not re-add training parallelism** — deleted 2026-07-21 after OpenMP deadlocks.
- **Do not set `SKIP_CV=1` in CI** — CV is the conformal calibration set, not a diagnostic.
- **Do not revisit CatBoost** — tested Jul 2026, degraded accuracy 18–20pp.
- **Do not rebuild the Reddit sentiment collector** — runner IPs get 403s and the features were
  refuted independently.
- **Do not replace with a fine-tuned LLM or a neural forecaster** (N-BEATS/PatchTST/TFT) — worse on
  numerical time series, slower, GPU-dependent.
- **Do not compare any accuracy figure across scorer versions.** Pre-2026-08-02 production DA came
  from a two-estimator scorer; CV, walkforward and production each score a different item universe.

---

## Files Reference

| File | Lines | Role |
|------|------|------|
| `backend/models/forecaster.py` | 5,543 | `ItemForecaster`: feature engineering, training, CV, predict |
| `backend/models/conformal.py` | 106 | Normalized split conformal band (sigma, q_hat, band). Pure numpy |
| `backend/models/steam_types.py` | 170 | Steam type field parser (rarity + weapon_type extraction) |
| `backend/models/item_parser.py` | 187 | Item-name parser **and** the phase-collapsed universe rule (`is_phase_collapsed`, `phase_collapsed_sql_filter`). No LightGBM import, so `api/` can use it |
| `backend/scripts/forecast_prices.py` | 381 | Entry point: retrain decision, train + predict, DB/Parquet write |
| `backend/scripts/backtest_accuracy.py` | 1,169 | Production backtest over stored forecasts |
| `backend/backtest/price_resolution.py` | 276 | Shared price estimator — both legs of the realised return |
| `backend/backtest/scoring.py` | 349 | Pure scorer: tiers, verdicts, cohort metrics, `MIN_FORECAST_DATES` |
| `backend/backtest/directional_test.py` | 269 | Pesaran–Timmermann headline: per-date excess, Newey–West t over dates, `PT_T_HURDLE = 3.0` |
| `backend/backtest/resolution_gate.py` | 285 | Unresolvable-rate gate (`MAX_UNRESOLVABLE_PCT = 10.0`) |
| `backend/backtest/walkforward_records.py` | 110 | Per-forecast record schema for paired offline comparison |
| `backend/backtest/paired_mde.py` | 86 | Paired minimum-detectable-effect for A/B arms |
| `backend/db/parquet.py` | 566 | `price-archive/ops/*.parquet` read/write; JSON-text nested columns |
| `backend/scripts/walkforward_backtest.py` | 625 | Fresh-model gate (`--max-items 500`, `STEP_DAYS = 60`) |
| `backend/scripts/evaluate_forecaster.py` | 354 | Legacy walk-forward evaluation (archived) |
| `backend/scripts/append_to_parquet.py` | 204 | Monthly/yearly archive partition writer |
| `backend/scripts/compute_mde.py` | — | Minimum detectable effect for the A/B harness |
| `backend/scripts/optuna_horizons_search.py` | — | Standalone per-horizon Optuna search |
| `backend/scripts/optuna_3d_search.py` | 161 | 3d horizon Optuna search (frozen winner lives here) |
| `backend/collectors/social_sentiment.py` | 332 | FinBERT ONNX INT8 sentiment scorer (workflow deleted; dormant) |
| `backend/tests/test_forecaster.py` | 2,297 | 154 forecaster tests |
| `backend/tests/test_minimal_model_shape.py` | 1,081 | Pins the 8-model shape and the removed code paths |
| `backend/tests/test_scale_free_features.py` | 185 | Price-scale invariance of every served feature |
| `backend/tests/test_degenerate_label_dates.py` | 164 | Snapshot-day and collector-cutover label voiding |
| `backend/tests/test_purged_production_split.py` | 104 | The production train/val purge band |
| `backend/tests/test_phase_collapsed_universe.py` | 224 | 19 cases pinning the Doppler exclusion at all four readers |
| `price-archive/item-metadata.parquet` | 109 KB | Rarity/weapon_type cache (computed, then dropped by the allowlist) |

Run tests with **`pytest tests`**, not bare `pytest` — `scripts/test_social_signal.py` imports
`thefuzz`, which is not in `requirements.txt`, and collection aborts.
