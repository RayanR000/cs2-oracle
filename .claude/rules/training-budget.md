---
paths:
  - "backend/models/forecaster.py"
  - "backend/scripts/{forecast_prices,evaluate_forecaster}.py"
---

# What a retrain actually sees

- **The budget and the price floor are one setting, and since 2026-08-08 production runs
  both.** `DEFAULT_TRAIN_FEATURE_ROWS = 1_200_000` and `DEFAULT_TRAIN_MIN_MEDIAN_PRICE = 1.0`
  (`scripts/forecast_prices.py`), mirrored on `ItemForecaster.train`'s own defaults so a bare
  `train()` is production's config. The floor restricts the universe by median price *before*
  `_stratified_item_subsample`, and at that budget the **926-item / 993,464-row ≥ $1 cohort**
  fits whole — **so the subsample never runs**. That is the entire justification: the
  subsample's hardcoded `seed=42` moves `mean_classifier_acc_ge1` by **sd 1.5–3.1pp** (8
  seeds, 2026-08-07), and no draw is the only setting that removes that variance rather than
  shrinking it. **It is not an accuracy claim** — the +3.50pp at 30d re-derives to +1.642pp
  [−0.809, +4.505], null (`docs/changelog/2026-08-08-per-fold-price-filter-rederived.md`).
  Changing one knob alone is the error: the budget without the floor spends 12x the
  wall-clock on the pool's tier mix, the floor without the budget leaves a smaller draw.
  `TRAIN_HORIZON_MAX_ROWS` was raised to 1.2M with them to stay non-binding — the per-horizon
  frame is 958,289 rows on this universe, so the old 700K would have bound. Escape hatch:
  `TRAIN_MIN_MEDIAN_PRICE=0`. An unparseable value keeps the floor, deliberately. The
  fresh-model gate still cannot detect any of this. The training **window** is a third knob:
  `TRAIN_DAYS_BACK` overrides the 4-year (1460-day) default (`forecaster.py::_build_production_split`, near the `TRAIN_DAYS_BACK` env read), and the
  sweep verdict is that 1yr is worst while 2/3/4yr are tied — so do not re-run it expecting a
  win. See
  `docs/changelog/2026-08-08-training-price-floor-shipped.md` and
  `docs/changelog/2026-08-07-training-item-universe.md`.
- **`TRAIN_PER_ITEM_ROWS` changes what the per-horizon cap spends the budget on.** Default
  off, where `_build_production_split` caps an oversized slice with a uniform
  `train_set.sample(n=max_rows)` and an item's share of the sample is its share of the
  rows. Set it and the cap draws an **equal quota per item** instead
  (`_per_item_row_sample`), which is the axis the paired harness measured: 71K rows/fold
  spread over 728 items beat 728K rows/fold at **+5.72pp vs +3.50pp at 30d**. Only
  `train_set` is thinned — never `val_set`. It is safe **only after feature engineering**:
  thinning earlier computes lags over a punctured series, and thinning before
  `prepare_targets` voids the label of any row whose `date + horizon` partner was dropped.
  It does **not** reduce feature-engineering time, which is what a wide
  `TRAIN_FEATURE_ROWS` actually buys.
- **`CV_DIAGNOSTIC_CLASSIFIER` decides which mid `q_hat` is centred on since 2026-08-11 — and
  that turns out NOT to move coverage. It stays a cost flag.** `predict` recentres the served
  mid on the classifier's call *after* building the band from `q_hat`, preserving both
  half-widths, so `q_hat` is fitted on a centre the serving path then moves;
  `_conformal_records(..., direction_class=...)` fixes the incoherence when this flag supplies
  an out-of-fold call, and `meta.json`'s `conformal_centre` records `"served"` or `"q50"` per
  horizon. **Measured 2026-08-11 across three arms at two anchors (runs `31529688893`,
  `31532517480`, `31532543867`): served coverage agrees within 1.1pp in all 8 cells, whichever
  centre and with the recentring on or off.** The reason is arithmetic — the displacement is
  bounded by `2*|mid|` and this model's predicted `|return|` is **median 0.95% / p90 9.01%**,
  against half-widths of `q_hat*sigma` with `q_hat` = 95.5 / 141.4 / 204.9 / 312.7 at
  3/7/14/30d, i.e. roughly ±10% to ±31%. So **do not pay the 932s for coverage** (52% of a
  classifier-on retrain, 872s off vs 1804s on locally, which would put a retrain at the
  30-minute cap), and **do not blame a low `IntCov` on the centre** — the replay reads ~81%
  pooled where the band is actually served. Never infer the centre from the environment; read
  `conformal_centre`, and read `BAND COVERAGE` in `scripts/replay_serving.py`.
  `docs/changelog/2026-08-11-conformal-centre-follows-serving.md`.
- **The calibration DENOMINATOR is incoherent with serving, and the fix is UNCONFIRMED (2026-08-12).** `q_hat` was
  fitted on the training label, which divides by the raw anchor quote, while `predict` quotes
  from the smoothed anchor and `resolve_anchors` scores against it — so the served band
  **over-covered at 87.2 / 91.8 / 90.6 / 89.0%** against an 80% target (19,917 prod outcomes;
  the half-width has to shrink to 0.72 / 0.61 / 0.73 / 0.77). `prepare_targets` now emits
  `calibration_target_col(h)` and `_conformal_records` can measure the residual from it, but the
  arm is **off by default** as `CONFORMAL_SERVED_BASIS=1` because it is **REFUTED as a remedy**:
  paired on one commit (arm `31564924194` vs control `31564943172`, same folds, identical `n`) it
  moves `q_hat` **UP at 4/4** — 94.72→103.70 / 141.77→148.80 / 204.34→205.80 / 312.05→316.96 —
  where the over-coverage needs it 25–39% smaller. The sign is itself a finding: the raw-basis
  residual can only be the smaller one if `r̂` already contains `p[d]/S[d]` and cancels part of
  it, i.e. **the booster is fitting the anchor deviation**. `sigma` is also exonerated (`p80(s)`
  below 1 in **19 of 20** strata; served `sigma` is 1.28/1.34/1.33/**0.93**× the calibration
  median but the band is wide across the whole range). ❌ **The expanding window is REFUTED too
  (2026-08-12, run `31611508808`).** `fold_q_hat` is now reported per fold, and the pooled value
  sits **0.94/0.91/0.92/0.84×** the p80 of the folds already at the 300K cap — *below* 1, where
  the hypothesis needs above. Dropping the one genuinely small-`n` fold **widens** the
  calibration to 1.06–1.13×, and at identical `n_train` the fold spread is still **1.56–2.25×**.
  Do not read the audit line's rho (−0.05/−0.14/−0.36/−0.48) as support: `CV_MAX_TRAIN_ROWS`
  binds from fold 4, leaving **4 distinct `n_train` values**, so the rho is fold 1's leverage —
  read `n_train_distinct` beside it. What the folds track is the **calibration window's
  volatility regime**, synchronised across horizons at Spearman 0.70–1.00, which is the same
  fact as the 58.2–99.2% per-date coverage. So the whole *"calibrate on different rows"* class
  is dead, including "use the late folds": a trailing 2–3 fold window is **worse** than pooled
  (1.13–1.21×). Only the single most recent fold moves the right way (0.88/0.84/0.72/0.64×) and
  it is an 18–23K-row estimate. The remedy class is a **conditional** `q_hat`.
  `docs/changelog/2026-08-12-expanding-window-refuted-for-band-width.md`.
  ❌ **And the TIME-VARYING half of that class is REFUTED too (2026-08-12) — the axis was wrong.**
  `docs/changelog/2026-08-12-the-band-is-tilted-in-sigma.md` scored five schemes over 731 dates
  offline (`backend/scripts/measure_conditional_qhat.py`, model-free, validated at
  **1.020/0.958/0.887/0.785×** the shipped `q_hat`). Level-matched to 80% marginal, a 60-day
  trailing `q_hat` and all three date-level state variables are **worse than pooled**, with
  placebos at ≤0.11pp — there is no date-level information to condition on, the same wall the
  market factor hit. **What carries the defect is `sigma`'s EXPONENT.** Level-matched coverage by
  `sigma` decile ramps **62→95%** at h=3 and **58→98%** at h=30, monotone in all ten deciles at all
  four horizons, and a fitted **β = 0.408 / 0.401 / 0.363 / 0.327** cuts the stratum error from
  **8.62/8.02/8.00/9.73pp to 0.57/1.20/1.82/1.01pp**. The mechanism needs no model: `sigma` spans
  11× across deciles while the `|residual|` it normalises spans 2.3–2.6×, so its range is ~4× too
  wide. **Not the clip** (1.2% floor / 1.0% cap; β moves 0.389→0.395 excluding every clipped row).
  ✅ **CONFIRMED on real OOF residuals, run `31619383780`** — elasticity **0.429 / 0.369 / 0.350 /
  0.313** on ~157K records, within **0.014–0.032** of the model-free prediction, and level-matched
  deciles ramping **62→93 / 60→95 / 57→96 / 52→97** (err 7.97 / 9.11 / 9.49 / 10.07pp). Read
  `cv_results[h].sigma_tilt` in `meta.json` or grep `Sigma-tilt audit`. ⚠️ **But read
  `stratum_err_pp_beta_heldout`, not the pooled err** — the pooled legs fit and score the exponent
  on the same rows. Held out (`β` fitted on all folds but the last), the error falls
  **7.26→1.78pp (−76%)** at 3d and **7.05→2.65pp (−62%)** at 7d but only **−26%** at 14d and
  **−12%** at 30d, where the corrected profile is not flat. **So a single global exponent is
  implementable at 3d/7d and NOT at 14d/30d** — those need a shrunk or non-parametric scale, which
  is a separate decision. ⚠️ **DISPUTED, and both proposed remedies are REFUTED (2026-08-12).**
  Walk-forward over 507–588 dates at the 14-day retrain cadence
  (`backend/scripts/design_sigma_scale.py`), shrinking β toward 1 is a **no-op** — its departure from
  1 is 6–14× the standard error of its own estimate, so λ = 0.996–0.999 and the shrunk arm reproduces
  the plain one to two decimals — and a non-parametric scale buys nothing outside 30d (a per-decile
  `q_hat` is worse at 3/4; a binned scale is better only inside the noise and worse on marginal
  coverage at 4/4). **One fitted exponent per horizon is the answer and it reaches all four**: the
  level-matched tilt falls **−89% / −94% / −84% / −73%** and the median band narrows to
  **0.87 / 0.86 / 0.84 / 0.77×**. That contradicts the −26% / −12% above, which scored **one**
  held-out fold where the long-horizon windows overlap; neither read dominates (this one is
  model-free, that one is real OOF residuals), so treat the reach as **unsettled until the paired
  dispatch** rather than settled either way. `fold_beta` is now reported per fold, which is the drift
  measurement the dispute turns on. ✅ **BUILT, gated off as `SIGMA_EXPONENT=1`** with a
  `sigma_exponent` input on `model-diagnostics.yml`
  (`docs/changelog/2026-08-12-sigma-exponent-implemented.md`,
  `docs/superpowers/specs/2026-08-12-sigma-exponent-design.md`). **Read this before touching it:
  `q_hat` and `conformal_beta` are a MATCHED PAIR** — `sigma` is ~0.07 so `sigma ** 0.4` is ~5×
  larger and `q_hat` absorbs it, measured at **833.43 → 151.18 (5.5×)** between the arms. A `q_hat`
  served at the wrong exponent is wrong by that factor, not partially corrected, so: never difference
  a `q_hat` across this flag (compare **coverage and WIDTH**), a missing `conformal_beta` means 1.0,
  and the served band reads it through `band_beta()` only. The **per-fold** `q_hat` is deliberately
  pinned at β = 1.0 even under the flag, so the published 0.94/0.91/0.92/0.84× series stays in one
  unit. ⚠️ It does **not** fix the marginal over-coverage: it closes the `sigma`-mix channel, worth
  **36–68%** of it, and on a calm period the corrected band covers **74–77%** — the level is open in
  BOTH directions and no 80% claim rests on it. ✅ **PAIRED-READ 2026-08-12 (`31629626929` /
  `31629638834`) and the flag STAYS OFF.** β reproduces (0.4241/0.3672/0.3370/0.3266) and the wiring
  is coherent, but served marginal coverage falls **−8.07/−8.26/−7.88/−2.19pp** over four anchors —
  a **regression at 3d**, where the control sat at 77.80% and the arm overshoots to 69.72%. **Ship β
  only with a re-level against the served `sigma` mix.** Two bar-reading rules come out of it: the
  0.87/0.86/0.84/0.77× width prediction **fails on the calibration set** (0.966/0.957/0.947/0.965)
  and holds only on the **served** band (0.794/0.768/0.750/0.761), so **name the basis before
  dispatching**; and `fold_beta` **hits the `[0.2, 1.0]` clamp** on early folds at 7d/14d/30d while
  drifting 0.20→0.61 across the fold sequence, so the pooled β sits below a 14-day refit's. The
  held-out leg again reaches 3d/7d only (−78%/−67% vs −22%/−35%), which settles the dispute against
  the walk-forward −84%/−73%. `docs/changelog/2026-08-12-sigma-exponent-paired-read.md`.
  ❌ **And it FAILS ITS OWN JOB at serving (`31643819235` / `31643829741`) — second, independent
  reason the flag is off.** The tilt is real where the band is served (control ramps **+15.66 /
  +9.94 / +8.10 / +9.13pp** across σ quintiles, 15 of 16 cells) but β **flips the sign** rather than
  flattening it (**−9.50 / −13.57 / −14.54 / −2.48**, 14 of 16 cells, t = −6.50/−8.49/−5.52/−3.27),
  and is **worse in magnitude than the control at 7d and 14d**. Read the **ramp**, never
  `sigma_tilt_pp`, which is not level-matched and is inflated by the arm's marginal drop.
  ⚠️ **The fitting population is the defect, and this REINSTATES the "refuted" prod read:** the
  served-flattening exponent interpolates to **~0.64/0.73/0.76/0.47**, double the OOF-fitted value
  and near the 0.798/0.692/1.034/1.150 taken on prod rows. OOF residuals and served outcomes are
  different populations — **do not carry an elasticity, a width or a coverage number across them**,
  which has now gone wrong twice in one day in opposite directions. The horizon story inverts too:
  OOF says 3d/7d work, serving says 3d/7d/14d overcorrect and 30d is right. Next instrument fits β
  on **resolved served outcomes** (a forecast-date panel, not a CV fold).
  `docs/changelog/2026-08-12-served-sigma-profile.md`.
  ⚠️ **A LEARNED scale exists as `LEARNED_SCALE=1` (off) and it does not fix this either.**
  `models/scale_model.py`: one small booster per fold on `log|residual|` from the item's features,
  cross-fitted so `q_hat` stays honest, **26.6s / 2.6% of training**. It and `SIGMA_EXPONENT` are
  **mutually exclusive and raise if both are set**. CQR was costed off run `31643819235` and
  **does not fit the 30-minute cap** — the OOF conformal CV is **646.8s of 1017.8s (63.5%)** and
  needs p10/p90 out of fold, i.e. **40.9–46.3 min** of training. Measured
  (`31649561391`/`31649571169`): beats the exponent at 7d/14d, loses at 30d, bands 0.78–0.88×.
  ⚠️ **Its 14d ramp of +0.65 is CANCELLATION, not flatness** (−4.09/−15.74/−8.95/**+31.39** per
  anchor); `2026-07-09` reverses it at 4/4 horizons and on ordinary anchors it overcorrects harder
  than the exponent at 3d/7d. 🔑 **Three scales, three different calibration→serving displacements,
  all calibrated to exactly 80% on their own records** — `sigma` over-covers (77.9/85.0/84.8/87.2),
  `σ**β` under-covers (69.7/76.7/76.9/84.9), learned under-covers (72.1/78.7/76.6/85.7). **The width
  variable is not the lever. Do not propose a fourth one.**
  `docs/changelog/2026-08-12-learned-band-scale-measured.md`.
  ✅ **The fourth one was proposed anyway and it WON — read this before repeating the "do not"
  above.** `CLIMATOLOGY_SCALE` replaces the `price_std_60d` sigma with the item's own trailing
  h-day return dispersion, shrunk toward its price tier's pool by `n_i/(n_i+K)`, passed to
  conformal as a `learned_scale` array. At matched 0.80 marginal coverage it is **43–47%
  narrower** than sigma at every horizon on 285 test dates, and unlike the sigma-exponent and
  learned-scale arms it wins **out of sample**, because it is variance reduction rather than
  signal extraction. **Default ON since 2026-08-19** (`CLIMATOLOGY_SCALE=1`; only the literal
  `"0"` disables, mirroring the training price floor), with `CLIMATOLOGY_SERVING_START =
  2026-08-20` marking the first clean climatology serve. It is mutually exclusive with
  `SIGMA_EXPONENT` / `LEARNED_SCALE` / `EXCEEDANCE_SCALE` — four alternative denominators, not
  layers; `_calibrate_conformal` raises if combined — and serving follows the artifact via
  `_climatology_scale_served`, so the cutover is atomic at the next retrain, not the moment the
  flag flips. ❌ The regime-reactive modifier `CLIMATOLOGY_REACTIVE` (a fast/slow EWMA ratio on
  `|return_1d|`, off by default) is **shelved**: it narrows ~20% uniformly but harms
  recently-calm/forward-volatile dates. `docs/changelog/2026-08-19-climatology-band-scale-default-on.md`,
  `docs/changelog/2026-08-20-climatology-reactive-band-scale.md`.
  ✅ **The band is also SIGNED now.** `conformal.calibrate_signed` returns two signed quantiles
  `(q_lo, q_hi)` and `band_signed` builds an asymmetric band around the mid; serving no longer
  recentres. `(q_lo, q_hi)` is a **matched set** with `beta` and the scale — never substitute one
  across a scale or beta boundary, and `band_signed(mid, sigma, -q_hat, q_hat)` is the only
  equivalent of a pre-signed artifact. `DIRECTION_UPWEIGHT` now defaults to **1.0** (neutral).
  `SIGNED_BAND_SERVING_START = 2026-08-19`; both start dates gate the served-outcome feedback
  calibration in `models/served_recalibration.py` so it never pools across band geometries.
  `docs/changelog/2026-08-19-signed-conformal-band.md`,
  `docs/changelog/2026-08-19-direction-upweight-neutral.md`. `q_hat` and every `fold_q_hat` are byte-identical to `31611508808`, so the
  audit moved no calibration. ⚠️ **The exponent does NOT explain the marginal over-coverage**
  (87.2/91.8/90.6/89.0% vs 80%): level-matching removes exactly that before the tilt is measured,
  and on the calibration set marginal coverage is 80% by construction. Six causes examined, the
  marginal defect belongs to none.
  `docs/changelog/2026-08-12-sigma-tilt-confirmed-on-oof-residuals.md`. ❌ Superseded and refuted at
  4/4, hardest where it said "fine": `d log|resid| / d log sigma` **0.798 / 0.692 / 1.034 / 1.150**,
  reconstructed as `half_pct / q_hat` from ~20K prod rows.
  Read `conformal_basis` in `meta.json` and `basis=` on the calibration line — never
  infer it from the environment. **Not** the
  `LABEL_SMOOTHED_ANCHOR` arm — the label is untouched. It was tested against the quiet-dates
  alternative and that was rejected: median `rel_cal` on the calibration window's own dates is
  1.01 / 0.96 / 1.07 / 0.97. **Per-date coverage still runs 58.2–99.2%** and no scalar `q_hat`
  fixes that. `docs/changelog/2026-08-12-conformal-basis-follows-serving.md`.
- **Training is fully sequential.** Horizons, quantiles, and ensemble members train one at
  a time; LightGBM's OpenMP threads supply the CPU parallelism. **Everything runs at
  `n_jobs = -1`** — the ensemble's `max(1, cpu_count // 2)` was deleted 2026-07-21 as
  leftover from the removed parallel-ensemble code; on a small runner it pinned the final
  fits to one thread while Optuna and the CV folds both took every core.
- **HP reuse is the steady state, and it is not free of served effects.** CI restores the
  model cache on every mode since 2026-08-10, so `reuse_hp` is true and Optuna is skipped
  (692.9s in run `31356483719`). Two things ride on that gate and must not be re-coupled to
  it: `full` needs `FORCE_RETRAIN=1` or the restored artifact trips the 14-day age gate and
  Monday trains nothing, and **regime models must still train** — `predict` prefers them over
  the global model, so skipping them changes the served mid rather than saving cost.
- **`NAIVE_INIT_SCORE=1` changes what the boosters PREDICT, not just what they see.** With it on,
  `-return_1d` is the `init_score` on every quantile `Dataset` and the boosters emit a **residual**
  to that baseline, so `predict`, the CV fold predictions, the Optuna objective,
  `_validate_feature_groups` and `_holdout_conformal_records` all add the offset back. Serving one
  of these boosters without it publishes a residual as a forecast, silently — which is why
  `_naive_init_score_served` follows `meta.json`'s `naive_init_score` and only falls back to the
  environment when no artifact is loaded. Training reads the environment, deliberately. The
  direction classifier is untouched. Read `rank_ic_edge`, bar `>= 0`; HP was selected against the
  un-offset target, so the run WARNs and any positive needs `FORCE_HP_SEARCH=1` to size. Off by
  default and **unmeasured**. See `docs/changelog/2026-08-10-naive-init-score-instrument.md`.
- **`FEATURE_NATIVE_NAN` is OFF in code and ON in production — read the artifact, not the
  default.** `os.environ.get("FEATURE_NATIVE_NAN") == "1"` defaults off (median imputation, the
  historical behaviour), but `.github/workflows/price-forecast.yml:228` sets it to `"1"`, so the
  nightly retrain trains with NaN passed through to the booster. Same matched-pair rule as
  `NAIVE_INIT_SCORE`: `_impute_features(served=True)` follows the artifact via
  `_feature_native_nan_served`, and train and serve must move together. What it removes is a
  bullish prior — the training cross-sectional medians of `return_180d/120d/90d` are ~6.3 / 4.2 /
  2.8, so an imputed short-history item was served a multi-month uptrend it never had
  (deep-model-review §10.4). The measured served effect is modest and **downward**
  (mean-reversion), the opposite of what the review predicted, so a retrain under it is a go/no-go
  read, not a known win. `docs/changelog/2026-08-21-feature-native-nan-built-gated-off.md`.
- **Size/speed levers are already documented.** See `docs/architecture/model-optimization.md`
  for the options that retain ≥90% quality — don't re-derive them.
