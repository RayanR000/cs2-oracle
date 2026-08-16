# Phase 2 implementation plan: exceedance-augmented conformal band scale

**Date:** 2026-08-16. **Depends on:**
`docs/changelog/2026-08-16-exceedance-probability-improves-band-conditional-coverage.md` (the offline
result: `sigma × √p_exceed` lifts volatile-tail coverage from 31–44% to 47–53% for ~5–15% more width,
against the real production scale). This plan turns that into a served, A/B-tested band scale.

## Goal and non-goal

- **Goal:** make the conformal band denominator `sigma × √p_exceed` an opt-in serving scale, retrain,
  and A/B it against the shipped plain-`sigma` band on **interval coverage + width**, with the
  conditional (by-|resid| decile) coverage as the primary read.
- **Non-goal:** any directional/trade claim. `p_exceed` is a magnitude signal; invariant 4 stands. No
  headline DA. The exceedance probability is used **only** as a band scale here.

## What already exists (integration seams, not greenfield)

- **A dormant learned-scale pathway.** `LEARNED_SCALE=1` fits a `scale_model` on `|residual|` and uses
  it as the band denominator: `_fit_learned_scale` (`forecaster.py:6942`), consumed by
  `_calibrate_conformal` (`:7055`, `q_hat = conformal.calibrate(resid, sigma, ALPHA, beta,
  learned_scale=learned)`, `:7099`), served via `band_scale` (`:6585`) into `conformal.band`
  (`:7769`, `learned_scale=self.band_scale(...)`). The shipped artifact has it **off**
  (`meta.json.learned_scale=None`, `conformal_beta=None` → β=1.0).
- **`resolve_scale` (`conformal.py:112`) already accepts a `learned` array as the full denominator**
  and **forbids `learned` + β≠1** (`:134`). β is already neutral in production, so `sigma × √p` can be
  passed straight through as `learned_scale` with no β conflict.
- **The direction classifier is the template for the exceedance head.** `_fit_direction_classifier`
  (`:6415`), stored in `self.direction_models` (`:873`), trained at `:5423`, served at `:7826`. The
  exceedance head mirrors it with `objective="binary"`.
- **Sigma is built once** at `forecaster.py:5162` (`price_std_60d/price`, clipped by `sigma_bounds`,
  frozen as `self.sigma_clip`). `price_cv_60d` is the same quantity and is a served feature.

Consequence: `sigma × √p_exceed` ships **as a `learned_scale` array**, reusing all of the
calibrate/serve plumbing. The only genuinely new artifact is the exceedance classifier.

## Tasks (each bounded; TDD per `superpowers:test-driven-development`)

**1. Exceedance label + head.**
- Add `target_exceed_{h}d` in `prepare_targets` (`:4225`), off the existing `target_return_{h}d`:
  `(target_return_{h}d > 100 * actionable_threshold(price_tier, "csfloat")).astype(int)` — one-sided,
  matching the validated screen. Inherits the existing ±500% winsorization and label voiding.
- Add `_fit_exceedance_classifier` mirroring `_fit_direction_classifier` (`:6415`): `objective="binary"`,
  `metric="binary_logloss"`, same feature matrix, same served-cohort (≥$1) reweighting, store in
  `self.exceedance_models: Dict[int, lgb.Booster]`. Train beside the direction head (`:5423`).
- Save/load beside `direction_models` (booster files + `meta.json` provenance flag `exceedance_scale`).
- Tests: label is one-sided and winsorization-safe; head trains and predicts a probability in (0,1);
  a horizon with <2 classes degrades gracefully.

**2. The scale, behind a flag `EXCEEDANCE_SCALE`.**
- New `exceedance_scale_enabled()` (env, default off), mutually exclusive with `LEARNED_SCALE` and
  `SIGMA_EXPONENT` (extend the guard at `:7077`/`resolve_scale`).
- Calibration: in `_calibrate_conformal` (`:7055`), when enabled, set
  `learned = sigma * sqrt(clip(p_calib, 1e-3, 1))` where `p_calib` = exceedance-head prediction on the
  **out-of-fold calibration rows** (same rows `_fit_learned_scale` uses), then the existing
  `conformal.calibrate(..., learned_scale=learned)` recomputes `q_hat` against it. **q_hat and the
  scale are a matched pair — do not reuse the plain-sigma q_hat.**
- Serving: extend `band_scale` (`:6585`) to return `sigma_arr * sqrt(clip(p_arr, 1e-3, 1))` when
  enabled, `p_arr` from the exceedance head on `latest_rows`. `conformal.band` already receives it via
  `learned_scale=self.band_scale(...)` (`:7769`); β stays neutral.
- Freeze nothing new in `sigma_clip`; `√p` needs no clip beyond the (1e-3,1) floor.
- Tests: `resolve_scale` rejects `EXCEEDANCE_SCALE` + β≠1; calibrated coverage on a synthetic frame is
  ~0.80; serving low≤mid≤high preserved (`_sanitize_forecasts`, `:7913`).

**3. Offline A/B — three arms, paired, on the end-anchored grid.**
- Arms: **baseline** (shipped plain sigma), **learned** (existing `LEARNED_SCALE=1` scale_model — the
  second adaptive baseline, never yet served), **exceedance** (`EXCEEDANCE_SCALE=1`).
- Harness: `scripts/replay_serving.py` (reports the **calibrated** `interval_coverage`) and/or a
  walkforward arm; follow `ab-statistics` (paired `walkforward_records`, `paired_mde`; win counts are
  **not** tests). Bump `VOTED_CACHE_VERSION` only if voting changes (it does not here).
- Metrics, quoted together: `interval_coverage` (calibrated) **and** `interval_coverage_dollar_basis`
  (published), **median width**, and the **primary read — conditional coverage by |resid| decile**
  (the 31–44%→47–53% tail-coverage lift is the thing being shipped, not marginal coverage).
- Run from the **controller** with a Bash waiter (retrain + walkforward is a long run;
  `[[long-runs-run-from-controller]]`), `mode=full` / `ALLOW_DRIFT_RETRAIN=1` to actually retrain.

**4. Decision + record.**
- Ship criterion, pre-registered: exceedance arm holds marginal `interval_coverage` within tolerance
  of baseline **and** improves conditional (tail-decile) coverage materially, at ≤~15% median-width
  cost — reproduced on the durable archive, paired, over ≥ the fold grid. A dated changelog either
  way.

## Validation gate (the bind)

Offline walkforward/CV can decide the scale. A **served headline** cannot be published until durable
≥$1 outcomes pass `MIN_FORECAST_DATES=20` (`scoring.py:145`) — ~0–4 dates exist per horizon today. So:
land the scale behind `EXCEEDANCE_SCALE`, prove it offline, and let served `interval_coverage`
accumulate before any published claim. This is a band-**calibration** change, not a DA claim, so
invariant 4 does not apply to the coverage metric — but do not quote any DA off `p_exceed`.

## Risks / rollback

- **Matched-pair footgun.** A q_hat calibrated at plain sigma served with the `√p` scale (or vice
  versa) mis-sizes the band ~massively (`conformal.py:300` warns the same for β). Mitigation: the flag
  gates calibrate and serve **together**; a test asserts the served scale identity matches the
  calibrated one via `meta.json`.
- **One-sided reallocation ceiling.** The offline result showed easy-bin coverage stays ~1.00 —
  `√p` widens hard items without narrowing easy ones, so expect width to rise, not fall. If the A/B
  shows width cost > tail-coverage benefit, prefer `√p` over `p` (already the recommended blend) or
  stop.
- **Rollback is the flag.** `EXCEEDANCE_SCALE=0` restores the shipped plain-sigma band byte-for-byte
  (β already neutral, learned already None).

## Estimate

Task 1 ~0.5 day, Task 2 ~0.5 day, Task 3 retrain+walkforward ~2–3h wall (controller-run), Task 4
~0.5 day. The classifier and the scale reuse existing patterns; the real cost is the paired A/B and
reading conditional coverage correctly.
</content>
