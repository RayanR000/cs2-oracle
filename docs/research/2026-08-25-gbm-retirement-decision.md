# What is left of the GBM, and the one test that would retire it

**Date:** 2026-08-25
**Type:** decision memo — no new measurement, consolidates four existing results
**Status:** open decision. Nothing here changes serving.

## Where the cutover already stands

The band the product actually sells has **two** parts, and only one of them is still GBM.

- **Width — already off the GBM.** `CLIMATOLOGY_SCALE=1` has been the served scale since
  2026-08-20 (`_climatology_scale_served`, `forecaster.py:1357`). Featureless per-item
  climatology is 33–46% narrower at matched 80% coverage and better calibrated
  (`2026-08-19-climatology-vs-gbm-band.md`).
- **Width, re-tested on a clean label — same verdict.** The 2026-08-22 re-run on the
  seam-free within-source label (`climatology_vs_gbm.py --label within-source`, no retrain)
  has the GBM losing at **every** horizon, climatology ~30% narrower vs ~25% on the dirty
  label. Cleaning the label confirms the result rather than rescuing the model.
- **Centre — still GBM, and the only remaining load-bearing use of it.** `r_hat` sets where
  the interval sits; the climatology scale sets how wide it is.

## The centre is weaker than its remaining role implies

Two results already bound it, neither of which was framed as a centre test:

1. The deep review measured median `|r_hat|` as moving the served interval by **<5% of its
   own width**. The centre barely moves the product.
2. The trade hunt closed with direction selection **null** — AUC 0.51–0.53 across three OOS
   windows, top-decile lift ≤1.0x, at every horizon including h=90/180. The GBM's directional
   ordering carries no usable information.

Null *ordering* is not the same as null *level* — a centre can be useless for ranking and
still beat a naive centre on average error. That gap is the whole of the open question.

## The test

Replay the served cohort and compare the GBM centre against a **last-price (random-walk)
centre**, holding the climatology scale fixed, on:

- centre MAE / median absolute error per horizon, and
- interval coverage and width with each centre, since a shifted centre changes coverage at
  fixed width.

This is the same shape as the width gate: read-only replay, no retrain, one script. The
existing `replay_serving.py` already reconstructs the served rows, so it is a flag on that
path rather than new infrastructure.

**Decision rule.** If the random-walk centre is within noise of the GBM centre on both, the
GBM is decorative end-to-end and the serving path collapses to climatology + last price —
which removes the retrain, the 28-feature set, and the artifact-matching machinery from the
daily chain. If the GBM centre wins on level, it stays, and the honest product description is
"a calibrated range with a small mean-reversion tilt."

## Why it is not running yet

Actions minutes are exhausted and all nine workflows are `disabled_manually` until the
billing reset. The replay is local-capable and does not need the chain; the retrain that
would follow a "retire" verdict does.
