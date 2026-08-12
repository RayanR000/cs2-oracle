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
  fresh-model gate still cannot detect any of this. See
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
  is a separate decision. `q_hat` and every `fold_q_hat` are byte-identical to `31611508808`, so the
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
- **Size/speed levers are already documented.** See `docs/architecture/model-optimization.md`
  for the options that retain ≥90% quality — don't re-derive them.
