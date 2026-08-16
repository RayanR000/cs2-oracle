# Exceedance band scale built and wired, gated off as EXCEEDANCE_SCALE=1

**Date:** 2026-08-16. Nothing shipped; the flag defaults off and no served band changes.
**Implements:** Phase 2 (Tasks 1–2) of
`docs/superpowers/plans/2026-08-16-exceedance-band-scale-phase2-plan.md`, off the offline result in
`docs/changelog/2026-08-16-exceedance-probability-improves-band-conditional-coverage.md`.

## What it does

`EXCEEDANCE_SCALE=1` makes the conformal band denominator `sigma * sqrt(p_exceed)` instead of
`sigma`, where `p_exceed` is a per-item binary head's probability the h-day move clears the
round-trip cost. `p_exceed` is a **magnitude** signal used only as a band-width scale — no
directional/DA claim rests on it (invariant 4 holds).

It rides the existing `learned_scale` plumbing: the exceedance scale is passed to
`conformal.calibrate` and `conformal.band` as a full denominator array, so `beta` stays neutral
(`resolve_scale` forbids a learned scale beside `beta != 1`) and `q_hat` is dimensionally tied to the
scale — a **matched pair**, exactly like `q_hat`/`beta` under `SIGMA_EXPONENT`. Mutually exclusive
with `LEARNED_SCALE` and `SIGMA_EXPONENT`; `_calibrate_conformal` raises if combined.

## Pieces

- **Head (Task 1).** `target_exceed_{h}d` in `prepare_targets` (one-sided,
  `target_return > 100 * actionable_threshold(tier, "csfloat")`, inherits the ±500% winsorization and
  every label void); `_fit_exceedance_classifier` (binary LGBM mirroring the direction head, served-
  cohort reweighting, `None` below two classes); trained per horizon in `_train_horizon_inline`,
  stored in `self.exceedance_models`, persisted as `exceed_clf_{h}d.txt`.
- **Scale (Task 2).** `exceedance_scale_enabled()` (env) + `_exceedance_scale_served()` (artifact-led).
  `_exceedance_learned_scale` builds `sigma * sqrt(clip(p, 1e-3, 1))` from the records' `exceed_p`
  column and takes precedence over `_fit_learned_scale` in `_calibrate_conformal`. `band_scale` serves
  the same shape from the served head. `meta.json` carries `exceedance_scale`.

## Design decision: the calibration probability is OUT-OF-FOLD, per fold

The plan said "`p_exceed` on the out-of-fold calibration rows" without pinning the mechanism. The trap
is the one `_fit_learned_scale` cross-fits around: the **served** exceedance head trains on the full
production split, which overlaps the CV calibration folds — so scoring calibration rows with the
served head is **in-sample**, shrinks the scale, makes `q_hat` too small, and under-covers in
production while looking flawless offline.

Resolution: `_cv_evaluate_horizon` fits a **per-fold** exceedance head on each fold's train and
predicts on its held-out val, and `_conformal_records` attaches that as a positional `exceed_p`
column (mirroring `direction_class=pred_cls`). The fold split already holds val out, so the honesty
is free — no separate cross-fit. Gated: no head is fit unless the flag is on. The single-holdout
fallback path scores in-sample from the served head, which is coherent with everything else that path
already does in-sample and warns about (it is unreachable in production).

## Matched-pair safety

Serving follows `_artifact_exceedance_scale` (from `meta.json`), not the environment: a plain-sigma
artifact reloaded with `EXCEEDANCE_SCALE=1` in the env keeps serving plain sigma, because its `q_hat`
was calibrated at plain sigma. A `q_hat` served against the wrong scale mis-sizes the band, so
`band_scale` refuses the scale unless the artifact says the calibration used it. Rollback is the flag:
`EXCEEDANCE_SCALE=0` restores the shipped band byte-for-byte.

## Not done here

- **No A/B (Task 3).** The paired offline walk-forward — `interval_coverage` + median width, primary
  read the conditional (|resid|-decile) coverage — is a controller-run retrain and has not run. No
  served coverage number exists.
- **Served headline stays blocked** by `MIN_FORECAST_DATES=20`; this is a band-calibration change, so
  the bind is on any published coverage claim, not on landing the flag.

Tests: `test_exceedance_{label,classifier,persistence,scale}.py` — label void-safety, head train/
predict, save/load round-trip, the flag, mutual exclusion, the matched calibrate/serve identity, meta
round-trip, and a source guard on the per-fold OOF wiring (the real integration proof is the Task 3
retrain, too slow for a unit test).
</content>
