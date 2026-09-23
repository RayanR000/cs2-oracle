# Model Architecture

> ⭐ **This is a RANGE (interval) forecaster (2026-08-15).** The served product is the conformal
> band and its center; the directional classifier described below is instrumentation, not the
> deliverable. **Directional accuracy is not a shippable claim** — direction is structurally
> unavailable at this data scale (~14–70 independent h-windows in the 2026 serving regime), and
> every relative / cross-sectional arm is CV-positive and serving-negative. The band over-covers:
> **85.90 / 88.70 / 86.46%** at h=3/7/14 against 80% nominal, median half-width
> **8.07 / 11.65 / 17.34%** (paired `replay_serving.py`, anchor 2026-06-16, n ≈ 1071, ≥$1 — one
> anchor, and the artifact post-dates it so the mid is leaky). Quote `interval_coverage`
> (calibrated) and `interval_coverage_dollar_basis` (published) **together**; never either alone.
> Do not propose another band-width scale (**four** have now been measured), a new row-set for
> `q_hat`, or `SERVE_OUTLIER_GATED_ANCHOR=1` as a coverage fix.
> `../changelog/2026-08-15-cs2-oracle-is-a-range-forecaster.md`.
>
> ⚠️ **The coverage/half-width figures above are as of 2026-08-15 and predate three shipped band
> changes**: the per-item climatology scale (2026-08-19, 33–46% narrower at matched coverage),
> the signed conformal quantiles (2026-08-19) and `DIRECTION_UPWEIGHT` returning to 1.0. Re-run
> `replay_serving.py` before quoting them.

> **⚠ There is no reportable production directional-accuracy number right now.**
> `MIN_FORECAST_DATES = 20` (`backtest/scoring.py:156`) refuses to publish a headline below 20
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

`ItemForecaster` (`backend/models/forecaster.py`) trains and serves **8 global LightGBM
models** — a median regressor and an exceedance head per horizon
(plus up to 3 × 4 regime q50 boosters on a cold retrain, see § Regime models):

```
3d horizon:  1 × q50 GBM  +  1 × binary exceedance head
7d horizon:  1 × q50 GBM  +  1 × binary exceedance head
14d horizon: 1 × q50 GBM  +  1 × binary exceedance head
30d horizon: 1 × q50 GBM  +  1 × binary exceedance head
```

Anomaly (`anomaly_clf_*.txt`) and ranking (`rank_*.txt`) heads are trained
alongside when `ANOMALY_GBM=1` / `RANKING_HEAD=1`. The 3-class directional
classifier is **offline-only since 2026-09-18**: routine training fits,
persists, restores and consults no direction booster
(`changelog/2026-09-18-multi-head-champion-challenger-built.md`).

The exceedance head is gated on `EXCEEDANCE_HEAD`, which is **off in the code default and set to
`1` in CI** (`price-forecast.yml:218`) — so a production artifact carries all 12 and a plain local
retrain carries 8.

`HORIZONS = [3, 7, 14, 30]` (:346), `QUANTILES = [0.5]` (:352), `N_ENSEMBLES = 1` with
`ENSEMBLE_SEEDS = [42]` and `ENSEMBLE_FEATURE_FRACTIONS = [0.7]` (:428-429). There is no
ensemble averaging and no p10/p90 model. The 40-model grid this replaced (36 global + regime
files) collapsed on 2026-08-05; see `docs/changelog/2026-08-04-minimal-model-results.md`.

**Division of labour:** the no-classifier fallback supplies the served
**direction** (threshold on the median return; the API discloses neutral
regardless); the q50 model supplies the **median**; conformal calibration over
the per-item climatology scale supplies the **band**;
the exceedance head supplies the disclosed `exceed_p` (published as `move_odds`, h3/h7 only —
`api/volatility_tags.py::CALIBRATED_MOVE_ODDS_HORIZONS`). It also produces a `confidence` tag,
which is stored and scored but **not published** — see "Confidence" below.

**Model version:** `lgbm-v3` (`scripts/forecast_prices.py:30`)
**Artifacts** in `backend/models/saved_models/`: `lgb_{horizon}d_q50_e0.txt` (see `_save_models`),
`lgb_{horizon}d_q50_{regime}_e0.txt`,
`exceed_clf_{horizon}d.txt`, `bias_corrections.json`, `meta.json`.
Routine training writes no `clf_{horizon}d.txt`; legacy files on disk are
ignored on load. `meta.json` carries a descriptive `components` manifest and
the per-horizon `centre_champions` mapping (initially all `gbm_q50`) — see
`changelog/2026-09-18-multi-head-champion-challenger-built.md`.
`MODEL_ARTIFACT_VERSION = 6` (:373) — `load_models()` raises `IncompatibleModelArtifact` on an
older cache rather than serving a band computed by a different scheme. v4 (2026-08-06) marks
the as-of lag lookup, which changes feature *values* and the persisted `feature_medians`
without changing the set; v5 (2026-08-06) marks the dollar-scale columns leaving the feature
set, after which a v4 booster's splits are thresholds in dollars and are meaningless; v6
(2026-08-09) marks Optuna selecting on within-date rank IC instead of early-stopped pinball
loss, so a cached `meta.json` cannot supply hyperparameters chosen under the old criterion.

### Directional classifier (offline-only since 2026-09-19)

Routine production training fits no direction booster. The reproducible
benchmark is `venv/bin/python -m scripts.direction_benchmark`
(production universe/features, fixed rounds, horizon+13 embargo); the
served fields come from the no-classifier fallback and the API discloses
neutral. What follows describes the retained offline instrument.

A per-horizon 3-class (down/flat/up) LightGBM trained on multiclass log-loss, which optimizes
the served metric directly. Labels come from a fixed ±`DIRECTION_FLAT_TOLERANCE_PCT = 0.5`
band on the realised return (`direction.py:18`). Movers (`|return| > tolerance`) are up-weighted 3.0×
(`DIRECTION_MOVER_WEIGHT_MAP`, :816) so the classifier spends capacity on the hard calls rather
than the large easily-predicted flat mass. Predicted class = `argmax`; confidence is `high` when
`max(prob) >= DIRECTION_CONFIDENCE_HIGH = 0.5` (:824).

`DIRECTION_UPWEIGHT` (:148) additionally scaled positive-return samples in
`_compute_sample_weights()` (see `_compute_sample_weights`). Set back to the neutral **1.0 on 2026-08-19**: at 1.5 it did
not correct direction (the sign comes from the classifier) but biased the q50 `|mid|` so the
served median sat at the ~58th–60th percentile. Env-overridable so the 1.5 control is
reproducible. `../changelog/2026-08-19-direction-upweight-neutral.md`.

**Vol-scaled labels are dead code.** `DIRECTION_VOL_MULTIPLIER_MAP` (:817), the `label_vol_30d`
column, and `_direction_threshold()` (see `_direction_threshold`) all exist, but every call site passes
`sigma_train=None, sigma_val=None`, so the threshold is always the fixed scalar. The
branch is unreachable in training and in CV.

### Conformal prediction band

`models/conformal.py` (501 lines) replaces the 24 p10/p90 quantile GBMs, which cost 223.2s of a
381.2s training budget for 39–48% empirical coverage against an 80% target.

⭐ **The band's per-item scale is a featureless climatology, not the GBM sigma, since
2026-08-19.** `CLIMATOLOGY_SCALE` defaults **on** (see `climatology_scale_enabled()`) and takes precedence over every other
scale in `band_scale()` (see `band_scale`). Per horizon it is a persisted
`{"table": {item_id: scale}, "tier_pool": {tier: scale}, "global": float}` built from the
training frame's own realised h-day return dispersion — no booster — with a James–Stein shrink
of a thin item toward its tier pool at `CLIMATOLOGY_SHRINK_K = 320`. It measured **33–46%
narrower at matched 80% coverage** than `sigma ** beta` and better-calibrated on served replay.
`../research/2026-08-19-climatology-vs-gbm-band.md`,
`../changelog/2026-08-19-climatology-band-scale-default-on.md`.

⭐ **The GBM band is decorative, and a clean label confirms it (2026-08-22).** The one
remaining objection to the climatology gate — that the GBM's loss was an artifact of a
seam-corrupted training label — was tested and refused. On a **seam-free within-source label**
(`scripts/check_label_seams.py::within_source_index`, re-run through
`scripts/climatology_vs_gbm.py --label within-source`, no retrain), climatology is **~30%
narrower** at matched coverage vs ~25% on the dirty label, and the GBM's OOS coverage *falls*
0.78 → 0.73 because the seams were inflating its `price_std_60d` scale. The GBM `sigma` band is
**instrumentation, not the deliverable**. This changed nothing in the serving path — climatology
was already the served scale. `../changelog/2026-08-22-gbm-band-decorative-on-clean-label.md`.
`CLIMATOLOGY_REACTIVE` (see `climatology_reactive_enabled()`, a multiplier off fast/slow EWMA vol) is a **shelved** modifier,
default off, after a prod A/B: `../changelog/2026-08-20-climatology-reactive-band-scale.md`.

| Step | Where | What |
|------|-------|------|
| Per-item scale (served) | `_fit_climatology_scale()` / `_climatology_scale_for_rows()` | Per-item h-day return dispersion, James–Stein-shrunk toward the tier pool. The default |
| Fallback scale | `conformal.sigma_from_columns` (`conformal.py:69`) | `sigma = price_std_60d / price` — the 60-day coefficient of variation. Used only where the climatology table has no entry, or with `CLIMATOLOGY_SCALE=0` |
| Clip bounds | `conformal.sigma_bounds` (`conformal.py:52`), called in `_train_horizon_inline` | 1st/99th percentile of the training frame's sigma distribution, frozen into `meta.json` |
| Calibration | `conformal.calibrate_signed` (`conformal.py:209`), called in `_calibrate_conformal` | `(q_lo, q_hi)` = the two **signed** tail quantiles of `residual / scale` over pooled out-of-fold CV residuals. `conformal.calibrate` (:178) still produces the symmetric `q_hat` as the fallback an artifact without the pair serves |
| Serving | `conformal.band_signed` (`conformal.py:483`), called in `predict()` | `mid + (q_lo·m)·scale` … `mid + (q_hi·m)·scale`, where `m = served_qhat_multiplier(h)` |

`NOMINAL_COVERAGE = 0.80` (`conformal.py:24`). The band is **no longer symmetric about the
median**: signed quantiles (live 2026-08-19,
`../changelog/2026-08-19-signed-conformal-band.md`) recentre it on the q50 residual's own
median, so an upward-biased q50 no longer forces a symmetric band inflated by its fat tail. An
artifact with no `conformal_q_lo`/`q_hi` pair falls back to `(-q_hat, +q_hat)` (`band_offsets`,
see `band_offsets()`) — the old band. Ordering is still guaranteed because `q_lo < q_hi` by construction, which
is why `predict()` does not repair quantile crossing. `_fix_quantile_crossing()` (see `_fix_quantile_crossing`) survives
only for `walkforward_backtest.py`'s baseline arm and the `ab_test_*` scripts;
`tests/test_minimal_model_shape.py` asserts it is absent from `predict()`. The band is
deliberately *not* floored at −100% in the band step, so `_sanitize_forecasts()` (see `_sanitize_forecasts`) is what
guarantees the served triple is ordered and positive.

**⚠️ `q_hat` is a MATCHED SET with `beta` and the scale.** A `q_hat` calibrated against the
climatology table is in that table's units; serving it against `sigma` gives an unrelated band,
not a degraded one. Every accessor (`band_scale`, `band_beta`, `band_offsets`) follows the
**artifact**, not the environment flag (`_climatology_scale_served`, see `_climatology_scale_served`), so a flag flip
cannot desynchronise serving from calibration.

`served_qhat_multiplier()` (see `served_qhat_multiplier`) is the served-outcome feedback correction — a scalar re-solved
on the served panel to the 80% nominal, clamped on read. It is **1.0 (no-op) on every artifact
today**: it self-activates only past the `MIN_FORECAST_DATES` gate.
`../changelog/2026-08-16-served-outcome-feedback-calibration-built-dormant.md`.

`predict()` raises rather than serving a forecast if `q_hat` is missing for a horizon.

**The residual `q_hat` is fitted on has its own column, and it is not the training label.**
`calibration_target_col(h)` → `target_return_{h}d_cal` divides by the **smoothed** anchor
`S[d]`, because that is what `predict` quotes from and what `resolve_anchors` scores against;
the label divides by the raw quote `p[d]`. Fitting `q_hat` on the label inflated it by the
anchor deviation `p[d]/S[d]` and the served band **over-covers at 87.2 / 91.8 / 90.6 / 89.0%**
against an 80% target. ❌ **The arm is OFF by default (`CONFORMAL_SERVED_BASIS=1`) and REFUTED as a
remedy**: paired on one commit it moves `q_hat` UP at 4/4 horizons (+9.5 / +5.0 / +0.7 / +1.6%),
where the over-coverage needs it 25–39% smaller. The basis is not the cause; the wrong sign is
evidence the booster is fitting `p[d]/S[d]`. `meta.json` carries `conformal_basis` per horizon and the calibration line logs it. This is the denominator half of what
`conformal_centre` fixed for the centre, and it is **not** the `LABEL_SMOOTHED_ANCHOR` arm —
`q_hat` is post-hoc, so it changes a band width and nothing the model learns.
⚠️ **It does not fix conditional coverage.** Per (horizon, date) the served band runs
**58.2%–99.2%**; `sigma` is a per-item trailing volatility with no date term, and ACI cannot
be validated until the panel has more than 1–7 forecast dates per horizon.
`docs/changelog/2026-08-12-conformal-basis-follows-serving.md`.

### Regime models

The code still supports per-regime ensembles (`REGIMES = ["bear", "range", "bull"]`, :419,
thresholds ±3% on `market_return_30d`), which at the current grid would be at most
3 × 4 = 12 extra models. A regime is skipped below `MIN_REGIME_TRAIN = 500` train rows (inside `_train_horizon_inline`).
⚠️ **An earlier version of this paragraph read "Nothing trains them in production: CI passes
`SKIP_REGIMES=1`, `meta.json` carries `trained_regimes: []`." That is wrong.** `price-forecast.yml`
never sets `SKIP_REGIMES`, and the Monday `mode=full` run is always cold (the model-cache restore
step is `if: mode == 'predict-only'`), so `_warm_retrain` is False and regimes *are* trained and
served. A cold retrain produces 8 regime boosters and `trained_regimes: ['bear','bull','range']`,
at **95.4s / 10.9%** of the retrain (measured 2026-08-09). `predict()` falls back to
`no regime models trained, using global` only for artifacts built with `SKIP_REGIMES=1` —
which is what the documented *local* retrain command passes, so a locally trained model and a
CI-trained one do not serve the same thing.

---

## Features

`_feature_group()` (:272-320) partitions engineered columns into thirteen groups:
`price_technicals`, `supply_depth`, `supply_churn`, `item_identity`, `item_metadata`,
`bymykel_metadata`, `tier_lead`, `temporal`, `events`, `cross_sectional`, `social`, `orderbook`, `other`.

**`FEATURE_GROUP_ALLOWLIST = ["price_technicals"]`** (:460, applied in `_filter_features`). This is the
single most consequential fact about the feature set: **every other group is computed on every
training row and then discarded.** Temporal, event, cross-sectional, rarity/identity, supply-depth
and social features all cost feature-engineering time and contribute nothing to any model. An
ablation (2026-07-24, purge-gap CV) found the 85 non-price features added no measurable
directional accuracy and hurt 3d/30d, so the columns stay computed but unused rather than being
deleted — the `ab_test_*` scripts build their own feature lists from the frame and need them.

Two further filters run before the allowlist:

- **`SHELVED_FEATURES`** (:579-627, unioned at :801) — **63** named columns withheld by name
  because most of them resolve to `price_technicals` and the allowlist would otherwise pass them
  straight through. Five groups:
  - **Six** volatility-asymmetry / oscillator-divergence primitives shelved 2026-07-31
    (`docs/changelog/2026-07-31-price-primitives-shelved.md`).
  - **Thirteen** volume features (`VOLUME_FEATURE_NAMES`, :636): the archive's `volume` column has
    been identically **0** since 2026-05 — stored as 0, never NULL, which defeats every guard in
    the volume feature code (`has_volume` tests `notna()` so it stays True, `volume_missing`
    reports "present") — so they carry real signal on pre-2026-05 training rows and are dead on
    100% of served rows. Re-measured and still shelved 2026-08-15
    (`docs/changelog/2026-08-15-volume-features-remeasured.md`).
  - **Five** dead-weight columns shelved 2026-08-18: `price_cv_20d`, `price_cv_30d`,
    `log_return_7d`, `autocorr_7d`, `rsi_missing`. They reached a booster but never ranked in any
    horizon's top-20 gain and are redundant or near-constant; a paired drop-5 ablation (400 items,
    24–25 folds, n≈180–190k) was **null at all four horizons** with every CI straddling zero. This
    is what takes the served set from 33 to 28.
  - **Thirty-seven** dollar-denominated columns, `_DOLLAR_SCALE_FEATURES` (:788-800), shelved
    2026-08-06: all `price_std_*`, `price_mean_*`, `price_min_*`, `price_max_*`, `price_lag_*`,
    plus `price_log`, the raw MACD trio and `bb_upper`/`bb_lower`. The target is a **percentage**
    return, so a dollar-scale input can only encode item identity — and on the 2026-08-06
    artifact these carried **55.6 / 70.2 / 77.5 / 86.6%** of total gain at 3/7/14/30d against a
    training median price of $0.086 and served items reaching $639. `price_tier` is deliberately
    kept: a bounded categorical is the honest way to express price level.
  - **Two** band-scale-only columns, `_REACTIVE_VOL_FEATURES` (:800) — `ewm_reactive_fast` /
    `ewm_reactive_slow`. Engineered so the `CLIMATOLOGY_REACTIVE` multiplier can read them, shelved
    so they never reach a booster.
- **Correlation pruning** at `PRUNE_CORRELATION_THRESHOLD = 0.95` (:837, applied in
  `_prune_features()`). `ALLOWLIST_BEFORE_PRUNE = True` (:467) runs the allowlist first, so
  the correlation matrix is built over the ~33 kept columns rather than 123.

All shelved columns are still **computed** — the conformal band reads `price_std_60d`,
`_apply_market_aggregates` reads `price_std_30d`, and the z-score / Bollinger / MA-distance /
support-resistance / log-return features are all derived from the means, mins, maxes and lags.

Net effect, all from `meta.json: horizon_feature_cols` and identical across all four horizons:
the shipped 2026-08-05 artifact carried **47** columns; shelving the volume features took the
2026-08-06 artifact to **36**; shelving the dollar columns took it to **33**; and shelving the
dead-weight five (2026-08-18) takes it to **28**, which is what the current artifact carries.
Each `MODEL_ARTIFACT_VERSION` bump exists to force the retrain rather than wait 14 days for the
age trigger.

`HORIZON_EXCLUDED_GROUPS` (:449-451) still excludes `cross_sectional` from 14d and
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
is *as-of* within `LAG_TOLERANCE_DAYS = 3` (:386): the aggregator drops whole calendar days
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

`train()` (see `train`) runs one horizon at a time via `_train_horizon_inline()` (see `_train_horizon_inline`).

### Training row budget

The most consequential knob in the system. Two separate budgets, previously conflated:

| Budget | Default | Where | What it bounds |
|--------|---------|-------|----------------|
| `max_feature_rows` | **1,200,000** | `DEFAULT_TRAIN_FEATURE_ROWS`, `forecast_prices.py:46`; env override `TRAIN_FEATURE_ROWS`, parsed at :149 | The frame **before** feature engineering — i.e. how many whole item histories the model ever sees |
| `max_rows` | **1,200,000** | `TRAIN_HORIZON_MAX_ROWS`, `forecast_prices.py:53` (passed at :536). `train()`'s own signature default is 300,000, which only a direct caller sees | Each horizon's slice **after** feature engineering |
| `min_median_price` | **1.0** | `DEFAULT_TRAIN_MIN_MEDIAN_PRICE`, `forecast_prices.py:58`; `_train_min_median_price()`, env `TRAIN_MIN_MEDIAN_PRICE` | Which items the budget may buy: a floor on each item's median price, applied **before** the subsample |

⚠️ The 100,000 / 700,000 / `None` triple this table used to carry is the **pre-2026-08-08**
default. The current defaults train on the whole ≥$1 cohort with **no subsample**, so the
subsample-seed paragraph below describes a path production no longer takes.

`_stratified_item_subsample()` (see `_stratified_item_subsample`) spends the first budget by selecting
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
**on determinism, not accuracy**: removing the draw removes the seed's 1.5–3.1pp variance.
It is **defaulted on** — both constants are the shipped defaults, not opt-ins. The
accuracy evidence is withdrawn — the stored +3.50pp at 30d re-derives to **+1.642pp
[−0.809, +4.505], null**
(`docs/changelog/2026-08-08-per-fold-price-filter-rederived.md`). See the changelog also for
the caveat that `predict()` still writes sub-$1 rows.

Raising it was measured and declined: 700,000 rows selects 646 items and costs **468.7s** against
**104.6s** at 100,000. That is more than the pre-rewrite 40-model grid cost — the
entire minimal-model saving spent to buy 12% of the item pool. The fresh-model gate cannot
resolve the accuracy difference either, so there is no measurement that would justify it. The two
budgets are deliberately not unified: one number would otherwise move coverage and the
per-horizon cap at the same time.

### Training window

`days_back=1460` (in `train()`, now overridable with `TRAIN_DAYS_BACK`), backfilled items only, read
from `price-archive/*.parquet` via DuckDB. The 2026 distribution-shift guard that used to
exclude the current year was removed once the May–June 2026 archive gap was backfilled.
A 1-vs-2-vs-3-vs-4-year sweep found 1yr worst and 2/3/4yr tied, so 1460 is not load-bearing
above ~730 (`docs/changelog/2026-08-18-training-breadth-is-accuracy-neutral.md`).

### Item universe

Four exclusions are applied at the read, not downstream, so training and `predict()` see the
same universe from the one query in `_fetch_voted_price_history()` (see `_fetch_voted_price_history`):

| Rule | Where | What it removes |
|---|---|---|
| `BID_SOURCES` | `models/item_parser.py`, dropped in `_apply_multi_source_voting` | `aggregator_buff163_buy` — a bid, which must not vote against asks |
| `TRAILING_WINDOW_SOURCES` | `models/item_parser.py`, dropped in `_apply_multi_source_voting` alongside `BID_SOURCES` (2026-08-09) | `aggregator_steam_7d/30d/90d` — Steam trailing-window MEAN sale prices, the wrong time basis rather than the wrong side of the book |
| `STEAM_SPOT_SOURCES` | `models/item_parser.py:89`, dropped in `_apply_multi_source_voting` alongside the two above (2026-08-17) | `aggregator_steam_spot` — Steam's fallback-free `last_24h`, which would cast a second Steam ballot beside `aggregator_sync` |
| `PHASE_COLLAPSED_SLUG_PATTERNS` | `models/item_parser.py:245`, applied as `phase_collapsed_sql_filter()` | Doppler / Gamma Doppler names, whose returns are phase-composition artifacts |

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
| Re-published snapshot (a day that is a byte copy of the previous one) | `_snapshot_dates()` (see `_snapshot_dates`) | `SNAPSHOT_DAY_FLAT_FRACTION = 0.99`, `MIN_DEGENERATE_CROSS_SECTION = 25` | Bad **endpoint** only — a copied day shifts no level, so it is harmless mid-window | 2 of 4,735 archive days (2026-07-16, 2026-07-22), both at 100.00%; next-highest day 69.01% |
| Collector cutover (a source-regime change in the stitched archive) | `_collection_shift_dates()` (see `_collection_shift_dates`) | `COLLECTION_SHIFT_FRACTION = 0.20` on the **item universe size** | Corrupts any label whose window **spans** it, so the whole horizon-wide anchor band is voided | 12 of 4,735 days (0.25%) — 4 in 2013, 1 in 2016, 7 in 2026 |

Cutovers are detected from the universe size and never from prices, deliberately: prices moving
cannot change how many items a collector returns, so the detector cannot mask a real crash
(`tests/test_degenerate_label_dates.py::test_a_price_crash_is_never_flagged`). The market-return
signature it catches is large — −31.6% on 2026-03-22, +17.4%/−17.8% on 2026-07-09/10, against
±0.5% on a normal day — and is a basis change, not a price move.

### Hyperparameter search

Optuna TPE with MedianPruner, per-quantile. `N_TRIALS_MAP = {3: 50, 7: 10, 14: 15, 30: 15}`
(:440) and `SKIP_HP_HORIZONS = [3]` (:444) — 3d is frozen on its 50-trial winner, warm-started
in `_optuna_search_params()` (see `_optuna_search_params`). 14d and 30d still search because they are the noisiest horizons;
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

Expanding-window CV: `CV_STEP_DAYS = 150` (:846), `VALIDATION_WINDOW_DAYS = 30` (:406),
`CV_MIN_TRAIN_DAYS = 200` (:847), each fold carrying a `horizon`-day purge gap, evaluated in
`_cv_evaluate_horizon()` (see `_cv_evaluate_horizon`) and capped per fold at `CV_MAX_TRAIN_ROWS = 300_000` (:891). A
1460-day frame yields 8–9 folds at the 100K config and ~33 at the shipped one. Folds report
persistence and momentum baselines and `edge_vs_best_baseline`, judged on the **classifier**
accuracy because that is the served signal; the ≥$1 cohort accuracy is reported alongside the
all-tiers number so it is comparable to the production headline.

CV is the calibration set, not just a diagnostic: `q_hat`/`(q_lo, q_hi)` and the confidence
thresholds are all fitted on pooled out-of-fold residuals (in `_calibrate_conformal`), in that order. This makes it the dominant
cost — **~148.6s of a 176.7s warm retrain, ~84%** — because it refits a median model per fold per
horizon purely to generate residuals.

**`SKIP_CV=1` is deliberately not set in CI** (`price-forecast.yml`, pinned by
`test_ci_workflow_does_not_skip_cv`; rationale at :6842-6854). It routes to a single-holdout
fallback that fits `q_hat` on the same rows used for early stopping and Optuna scoring, so the
band under-covers; `train()` logs that at WARNING. It survives as a local/dispatch
speedup only.

### Boosting

`BOOSTING_TYPE = "gbdt"` (:439) for all four horizons. There is no per-horizon boosting map and
no DART: when DART was finally measured against GBDT on a trustworthy gate, 14d **improved**
+3.14pp (CI [+1.955, +4.373]) and 30d was unchanged — and 14d was the horizon DART was supposedly
earning its cost on. The dropout branches went with it.

`data_sample_strategy` is `bagging` for the median model (see `_base_params`); GOSS was reverted 2026-07-29
after it measured worse under the quantile objective's constant ±alpha gradients.

### Parallelism

**None.** Horizons, quantiles and ensemble members train sequentially; LightGBM's OpenMP threads
supply the CPU parallelism, with ensemble members getting `n_jobs = cpu_count // 2` and the
Optuna search params keeping `n_jobs: -1`. Both the horizon `spawn` Pool and the ensemble
`ThreadPoolExecutor` were deleted on 2026-07-21 (−209 lines) because they deadlocked under
OpenMP and the surrounding timeouts were masking it. See
`docs/changelog/2026-07-21-remove-training-parallelism.md`. **Do not re-add this.**

`_gpu_available()` (:220) still probes CUDA in a subprocess — required because
`lgb.train(device="cuda")` segfaults uncatchably on a CPU-only pip wheel — and sets
`device: cuda|cpu` in `_train_horizon_inline()`. Every shipped model in `meta.json` carries `device: cpu`.

### Deleted layers

- **Residual stacking (Ridge on LightGBM residuals)** — deleted 2026-07-25. It was fit on raw
  unscaled feature values, so it extrapolated without bound: a penny item with a legitimate +900%
  `return_Nd` got a correction in the +100,000% range, over-correcting 99% of 14d items and
  inverting quantile ordering on 100% of predictions. Tombstone comment at :825. No `.pkl`
  artifacts exist and `scikit-learn` is not in `requirements.txt`.
- **CatBoost** — removed; not a dependency. The stale `backend/catboost_info/` directory it left
  behind was deleted 2026-08-10.
- **Momentum fallback** — `MOMENTUM_FALLBACK_HORIZONS = []` (:806), superseded by the classifier,
  which beats momentum at every horizon including 30d. `_recenter_on_momentum()` (see `_recenter_on_momentum`) survives
  as an unreachable path behind that empty list.

### Retrain triggering

**Age-based only.** `RETRAIN_INTERVAL_DAYS` defaults to **14** (`forecast_prices.py:476`) and a
retrain fires when the artifact's age reaches it, or when `FORCE_RETRAIN=1` (:475).

Drift is **report-only**. `DRIFT_DA_THRESHOLD = 60.0` (`forecaster.py:397`) is now an alert threshold: the old
drift-triggered retrain compared measured DA against a 60% floor the model has never reached, so
it fired on *every* run and added a measured 465s to an 835s daily step, serving forecasts from a
throwaway warm retrain instead of the scheduled model. Set `ALLOW_DRIFT_RETRAIN=1`
(`forecast_prices.py:513-529`) to restore the old behaviour.

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

`predict()` processes items in chunks of `PREDICT_CHUNK_ITEMS` (default 1000, see `predict`) and reuses a
3-day-TTL engineered feature cache (`ENGINEERED_CACHE_VERSION = 4`, :996). The predict frame is
first truncated to `PREDICT_TAIL_ITEM_DAYS = 240` observed item-days (see `predict`).

### Eligibility

`PREDICT_MIN_HISTORY_DAYS = 14` distinct days in the Parquet archive (:378, applied in `predict()`) —
looser than the training floor of `MIN_HISTORY_DAYS = 30` (:374) because the live aggregator
series is still young. Restricted to the `is_backfilled` **serve** universe read from the DB
(`_resolve_backfilled_slugs()`, see `_resolve_backfilled_slugs`) — a flag re-derived from the archive on every
`init_local_db.py` run rather than set by hand: **~5,542 items** as of 2026-08-06. The narrower
**train** universe is derived straight from the archive and never read from the DB.

### Serving transform

Features come from the last row per item, reindexed with `fill_value=0` for absent columns and
then median-filled from `meta.json: feature_medians` (via `_impute_features()`, see `_impute_features`).

### Spike smoothing

The base price used to convert percentage returns into dollars is a span-bounded 3-observation
median near the anchor (`_smoothed_anchor_prices()`, see `_smoothed_anchor_prices`, called in `predict()`), matching the backtest's resolver
rather than "the last three rows on file". Items whose latest price deviates >10% from that
median are logged.

### Serving order

Order matters and is asserted by tests:

1. q50 predicts the median return; the horizon is skipped unless `0.5 in preds` (in `predict()`).
2. `conformal.band_signed(mid, scale, q_lo·m, q_hi·m)` builds the interval (in `predict()`), where the
   scale comes from `band_scale()` (the climatology table by default) and `m` is
   `served_qhat_multiplier`. `sigma_arr` is computed once outside the horizon loop because it does
   not vary by horizon (`_sigma_for_rows()`, called in `predict()`).
3. The classifier predicts class probabilities (in `predict()`).
4. `_blend_returns_with_prior` blends with the previous day's forecast at
   `FORECAST_BLEND_WEIGHT = 0.15` (:836, applied in `predict()`) to damp direction flip-flopping.
5. Per-tier bias correction — threshold-based, with the additive correction only as a fallback
   when no threshold data exists (in `predict()`). `BIAS_FIT_SCHEMA_VERSION = 2` (:822) discards
   thresholds fitted without a date-coverage guard.
6. The exceedance head emits `exceed_p` (in `predict()`) — served whenever a head is in the artifact,
   independent of which scale the band used.
7. `_sanitize_forecasts()` (see `_sanitize_forecasts`) clamps NaN/INF/negative prices to `current_price` with `flat`
   direction and `low` confidence, and downgrades high confidence on zero-volume items.

> ⭐ **`_recenter_on_direction` is no longer in the serving path (RANGE STANCE, 2026-08-19).** It
> used to move the median onto the classifier's call last, which put the mid somewhere the band's
> coverage had not been fitted and made a range forecaster behave like a directional one. The
> band's skew now comes from the signed conformal offsets, calibrated on the q50 residual itself.
> The classifier's call still populates the `direction` / `confidence` fields; it no longer moves
> the price. `_recenter_on_direction()` (see `_recenter_on_direction`) survives for the CV diagnostic.
> `docs/specs/2026-08-19-signed-conformal-quantile-design.md`.

> ⚠️ **The no-classifier fallback branch is a live hazard, measured 2026-08-11.** When
> `self.direction_models` has no entry for a horizon, `predict()`'s `else` branch derives direction from `mid_ret` against `t_down` / `t_up`, **defaulting to
> ±`DIRECTION_FLAT_TOLERANCE_PCT` (±0.5%)**. Against a served mid whose `|mid_ret|` distribution is
> shrunk far harder than realised returns, that dead band swallows the majority of calls: on
> 2026-07-19 — the last date served before a classifier existed — it produced `flat` on **64.0%** of
> ≥$1 items at h=3 against a 23.1% realised flat rate, and cost ≥ +8.7pp of DA against a plain
> zero-threshold sign rule. Flat is 0.0% on every classifier-era ≥$1 date, so nothing served today
> is affected. The branch no longer fires silently: `_warn_no_classifier()` (see `_warn_no_classifier`)
> logs a WARNING with the fallback and flat-call counts, called in `predict()`.
> `docs/changelog/2026-08-11-the-da-gap-is-the-market-direction-of-five-dates.md`.

### Confidence — computed and stored, NOT published

Binary `high` / `low`, taken from the classifier's max class probability against
`DIRECTION_CONFIDENCE_HIGH = 0.5`. `_compute_confidence()` (see `_compute_confidence`) and the
per-horizon `confidence_thresholds` in `meta.json` are only reached on the no-classifier
fallback path.

> ❌ **Withdrawn from the API 2026-08-12.** The cut was never calibrated against outcomes.
> Within-date on the ≥$1 `lgbm-v3%` panel, on the 11 cells with `n_high >= 30`, the `high`
> cohort's own directional accuracy is **29.0–45.8%** — every cell below a coin flip against
> `CONFIDENCE_TARGET_ACCURACY = 80.0` — while its gap to `low` runs −8.26 to +4.65pp with mixed
> sign. It orders nothing and its label is false. `PredictionOut` and `TrendAnalysisOut` no
> longer carry it; the column, the writer and `conf_gap_pp` stay so the withdrawal is
> falsifiable. **The backtest's pooled `conf_gap_pp` is not the evidence** — it splits over the
> whole record set, so it inherits the market-composition term, and its −38.7pp at h=30 is one
> item. `docs/changelog/2026-08-12-served-confidence-withdrawn.md`.

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
`MAX_UNRESOLVABLE_PCT = 10.0` (`backtest/resolution_gate.py:118`) fails the run rather than
reporting a cohort riddled with guaranteed misses. Aggregates land in `prediction_accuracy` (per
price tier, ≥$1 headline), per-forecast outcomes in `forecast_outcomes`.

Tier 0 (<$1) is reported separately rather than folded into the headline: it is ~72% of the
evaluated universe, and at those prices one cent is a 20% move, so the direction label is
dominated by tick quantisation. `api/serving_policy.py:37` sets `MIN_SERVED_PRICE_USD = 1.0` equal to
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

> ⚠️ **And `DA − realised_down_rate` is not readable on a handful of dates either (2026-08-11;
> figures corrected 2026-08-12 against prod Postgres).**
> Decomposed per (horizon, forecast date) on the ≥$1 `lgbm-v3*` panel, **52% / 82% / 103%** of the
> apparent **−11.89 / −12.08 / −15.54pp** production deficit at 3/7/14d is the realised direction of
> the 3–8 dates in the panel; the within-date term is **−2.26 / +0.34 / +4.98pp** — positive at
> h=14. The CV-basis leg (+2.70 / +0.17 / +0.42pp) was **not recomputed and is unverified**, so do
> not cite it as agreement with the +3.5 / +0.1 / −1.3 above.
> Per-date sd of the excess is **14.96 / 22.40 / 21.56 / 24.39pp**. **Do not quote pooled excess on
> fewer than ~50 forecast dates**; the per-date null the PT
> block estimates below is the quantity that survives a short panel. And **do not read the local
> working copy of `price-archive/ops/forecast_outcomes.parquet` for a panel figure** — it held 14,668
> of the 23,073 rows with verdict-selected gaps, which is what cost the superseded numbers above.
> **Neither Parquet copy is the full panel** (the durable archive is fresh and cell-complete but only
> 10 dates deep, so the publish leg is fine); query prod Postgres read-only.
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
- **Four band-width levers are measured dead — do not re-propose them.** Listing count
  (`log1p(listing_count)` as a width conditioner, 0/3 horizons,
  `../changelog/2026-08-18-listing-count-conditioner-refuted.md`), iflow/Skinport volume in the
  scale (net negative, `../changelog/2026-08-17-volume-in-scale-is-net-negative.md`), recency
  weighting (`SAMPLE_WEIGHT_HALFLIFE_DAYS = 0.0`, forecaster.py:192; did not transfer to served coverage,
  `../changelog/2026-08-14-recency-decay-does-not-transfer-to-served-coverage.md`), and the
  reactive climatology multiplier (`CLIMATOLOGY_REACTIVE`, shelved after a prod A/B,
  `../changelog/2026-08-20-climatology-reactive-band-scale.md`). The exceedance scale was also
  refuted as a band scale (`../changelog/2026-08-16-exceedance-band-scale-refuted-at-serving.md`)
  while surviving as the disclosed `exceed_p`. The width variable is not the lever.
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
  bump `VOTED_CACHE_VERSION` when voting or the DuckDB query changes. Now at **v8** (:1025): v2
  marked the `BID_SOURCES` exclusion, v3 the phase-collapsed names leaving the universe, v4 the
  phantom slug keys, v5 added `n_ask_sources`, v6 excluded `TRAILING_WINDOW_SOURCES` (Steam's
  trailing-window means), v7 excluded `STEAM_SPOT_SOURCES` (`aggregator_steam_spot`). See `.claude/rules/item-universe.md` for the full history and the
  cache-bump trigger.
- **The `ab_test_*` harnesses do not share the production universe.** Ten-plus of them carry
  private archive globs and filter neither `BID_SOURCES` nor the phase-collapsed names, so they
  train on a universe production no longer has. Tracked as step 5 of
  `docs/research/2026-08-07-next-steps.md`, alongside their missing purge and embargo.
- **Dead code that survives deliberately:** `_fix_quantile_crossing` (backtest baseline arm),
  `_recenter_on_momentum` (empty `MOMENTUM_FALLBACK_HORIZONS`), the vol-scaled direction-label
  branch (all call sites pass `sigma=None`), `_recenter_on_direction` on the serving path
  (retired 2026-08-19; still used by the CV diagnostic), and every non-`price_technicals`
  feature group (computed, then dropped by the allowlist). **Regime training is NOT dead code** —
  CI never sets `SKIP_REGIMES`, and the Monday `mode=full` run is always cold, so regimes are
  trained and served; see § Regime models.

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
| `backend/models/forecaster.py` | 12,152 | `ItemForecaster`: feature engineering, training, CV, predict |
| `backend/models/conformal.py` | 501 | Split conformal band: `sigma_bounds`, `calibrate`/`calibrate_signed`, `band`/`band_signed`, `fit_beta`, `resolve_scale`. Pure numpy |
| `backend/models/steam_types.py` | 178 | Steam type field parser (rarity + weapon_type extraction) |
| `backend/models/item_parser.py` | 400 | Item-name parser **and** the phase-collapsed universe rule (`is_phase_collapsed`, `phase_collapsed_sql_filter`). No LightGBM import, so `api/` can use it |
| `backend/scripts/forecast_prices.py` | 715 | Entry point: retrain decision, train + predict, DB/Parquet write |
| `backend/scripts/backtest_accuracy.py` | 1,518 | Production backtest over stored forecasts |
| `backend/backtest/price_resolution.py` | 245 | Shared price estimator — both legs of the realised return |
| `backend/backtest/scoring.py` | 630 | Pure scorer: tiers, verdicts, cohort metrics, `MIN_FORECAST_DATES` |
| `backend/backtest/directional_test.py` | 269 | Pesaran–Timmermann headline: per-date excess, Newey–West t over dates, `PT_T_HURDLE = 3.0` |
| `backend/backtest/resolution_gate.py` | 406 | Unresolvable-rate gate (`MAX_UNRESOLVABLE_PCT = 10.0`) |
| `backend/backtest/walkforward_records.py` | 240 | Per-forecast record schema for paired offline comparison |
| `backend/backtest/paired_mde.py` | 291 | Paired minimum-detectable-effect for A/B arms |
| `backend/db/parquet.py` | 549 | `price-archive/ops/*.parquet` read/write; JSON-text nested columns |
| `backend/scripts/walkforward_backtest.py` | 770 | Fresh-model gate (`--max-items 500`, `STEP_DAYS = 60`) |
| `backend/scripts/append_to_parquet.py` | 285 | Monthly/yearly archive partition writer |
| `backend/scripts/compute_mde.py` | 74 | Minimum detectable effect for the A/B harness |
| `backend/api/volatility_tags.py` | — | `swing_pct` / `move_odds` / `stability_label` derivation; `CALIBRATED_MOVE_ODDS_HORIZONS = (3, 7)` |
| `backend/models/served_recalibration.py` | — | The served-outcome `q_hat` feedback factor and its clamps |
| `backend/collectors/social_sentiment.py` | 345 | FinBERT ONNX INT8 sentiment scorer (workflow deleted; dormant) |
| `backend/tests/test_forecaster.py` | 2,682 | Forecaster unit tests |
| `backend/tests/test_minimal_model_shape.py` | 1,780 | Pins the minimal-model shape and the removed code paths |
| `backend/tests/test_scale_free_features.py` | 192 | Price-scale invariance of every served feature |
| `backend/tests/test_degenerate_label_dates.py` | 158 | Snapshot-day and collector-cutover label voiding |
| `backend/tests/test_purged_production_split.py` | 105 | The production train/val purge band |
| `backend/tests/test_phase_collapsed_universe.py` | 227 | Pins the Doppler exclusion at all four readers |
| `price-archive/item-metadata.parquet` | ~0.1 MB | Rarity/weapon_type cache (computed, then dropped by the allowlist) |

`evaluate_forecaster.py`, `optuna_horizons_search.py` and `optuna_3d_search.py` used to be listed
here and **no longer exist**; the frozen 3d hyperparameters live in `_optuna_search_params`.

Run tests as `venv/bin/python -m pytest tests/test_<name>.py -q`. A bare `pytest` collects
cleanly since 2026-08-10 (the `thefuzz`-importing `scripts/test_social_signal.py` was deleted),
but the full suite trains models — prefer targeted runs.
