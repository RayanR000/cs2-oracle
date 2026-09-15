# Pre-registration: q_hat fold-design dispersion probe

**Date:** 2026-09-14, written and committed **before any probe number is computed.**
**Status:** PROPOSED — runnable now, from the controller (not a subagent).
**Orders:** `docs/specs/2026-09-14-wait-window-workplan.md` Item 3.
**Follows:** `changelog/2026-09-13-qhat-bagging-mondrian-probe.md` (variance
confirmed 1.11–1.39x, bagging null, Mondrian void on placebo),
`changelog/2026-09-13-mondrian-oof-confirm-closed.md` (real-OOF per-fold spread
3.4–3.6x), and the vol-rank lesson (tuning against a single split manufactures
"6–8% narrower" claims that later refute).

## What this is and is not

The served band over-covers ~2.2x and the served-outcome feedback (Item 1) will
correct the LEVEL once the 20-date gate clears. This probe targets the
**variance of the estimator**: per-fold `q_hat` swings ±26% on fold composition
alone (h=3, folds 4–9: 68.8 → 126.9 at identical 300K training rows), and
`spearman(n_train, q_hat)` flips sign by horizon (−0.66/+0.22/+0.76/−0.52), so
training size explains approximately none of it.

This is **not** a conditioning experiment. No sigma, no climatology modifier, no
Mondrian bin, no bagging, no per-item anything — that family is 0-for-7+
(`band-width-levers-refuted`). The only lever is the fold **design**: stride,
count, and overlap. A null changes nothing (flat pooling stays); a positive
changes only the CV grid, never the band formula.

## Harness

New script `scripts/measure_qhat_dispersion.py` (to be built): full retrains on
the existing archive panel with `CV_STEP_DAYS` overridden via env
(`forecaster.py:820` already supports it; default 150, ~7 non-overlapping
folds), reading per-fold `fold_q_hat` out of `cv_metrics` exactly as
`confirm_mondrian_oof.py:146-154` does. No serving change, no artifact publish.
Horizons never pooled. Report per horizon: per-fold q_hats, max/min ratio, and
coefficient of variation.

Arms (each a full retrain; conformal CV is 57% of an 847s retrain, so run from
the controller with a Bash waiter):
- **Control:** `CV_STEP_DAYS=150` (status quo), seed A.
- **Placebo:** `CV_STEP_DAYS=150`, seed B (different `TRAIN_SEED`/sampling seed,
  identical grid). Measures the sampling-noise floor: dispersion that survives
  here is seed noise, not design signal.
- **Dense:** `CV_STEP_DAYS=75` (more folds, more overlap).
- **Sparse:** `CV_STEP_DAYS=300` (fewer folds, less overlap).

One grid question, one answer per horizon. No further arms without a second
pre-registration — that is how single-split tuning happens.

## Falsifiable bar for "less noisy" (fixed before reading numbers)

A fold design wins iff ALL of these hold:
1. Its per-horizon q_hat CV is **≥30% below control at ≥3 of 4 horizons**,
   with no horizon worse than **+10%** vs control.
2. The control-vs-placebo CV gap is **<15%** at every horizon (if the grid
   itself is seed-unstable, nobody wins — the probe is void, not null).
3. Matched-width coverage on the prod-basis replay (`replay_serving.py`) is
   within **±1pp** of control at every horizon — dispersion must not be bought
   with a narrower, worse-covering band.

If (1) fails everywhere: the estimator variance is irreducible by grid choice;
close Item 3 and serve the feedback multiplier on the noisy width as planned.
If (2) fails: re-run control/placebo once; if still unstable, void and report.
