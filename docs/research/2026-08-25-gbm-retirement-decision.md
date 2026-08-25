# What is left of the GBM, and the one test that would retire it

**Date:** 2026-08-25
**Type:** decision memo — no new measurement, consolidates four existing results
**Status:** MEASURED 2026-08-25. The centre loses to a random walk. Nothing here changes serving yet.
**Script:** `backend/scripts/centre_vs_lastprice.py` (read-only; no artifact, no retrain)

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

## Result

`centre_vs_lastprice.py`, run on the canonical archive's 32,603 resolved served
rows (>=$1, `excluded_forecast_date` applied):

| h | rows | dates | MAE_gbm | MAE_naive | skill | 90% CI | cov_gbm | cov_naive |
|---|------|-------|---------|-----------|-------|--------|---------|-----------|
| 3  | 11,160 | 11 | 0.03259 | 0.03076 | **-0.059** | [-0.079, -0.040] | 0.916 | 0.921 |
| 7  | 11,681 | 11 | 0.04846 | 0.04557 | **-0.063** | [-0.107, -0.032] | 0.914 | 0.921 |
| 14 | 7,779  | 8  | 0.08773 | 0.07886 | **-0.113** | [-0.175, -0.052] | 0.835 | 0.853 |
| 30 | 1,983  | 2  | 0.13686 | 0.14375 | +0.048 | [+0.015, +0.078] | 0.770 | 0.718 |

skill = 1 - MAE_gbm/MAE_naive; positive means the GBM beats the random walk.

**The GBM centre is worse than quoting today's price at h=3, 7 and 14** — 6-11%
more absolute error, with the date-bootstrap CI clear of zero at all three. It
is negative on **25 of the 27 date-horizon cells** individually, so this is not
one bad date carrying a pooled number. Coverage at identical width is a wash
(91.6% vs 92.1%), which is the <5%-of-width finding showing up from the other
side: the centre is too small to help the band and just large enough to hurt
the point estimate.

The h=30 win is the one positive cell and should not be believed yet: two clean
dates, both inside the same July drawdown.

**Not a trend artifact.** The panel is a falling market throughout (mean realised
r of -0.001 to -0.068 by date) while mean `r_hat` sits at +-0.003 for most dates.
The naive centre is not winning because it happened to bet the trend — it wins
because `r_hat` is near-zero noise added on top of it. The two h=14 dates where
the model leaned meaningfully down (mean `r_hat` -0.021, -0.013) are also its
two best cells, which is consistent with mean-reversion being the only real
signal in the centre and it being too weakly expressed to pay for itself.

**Caveat, stated in the script's own output:** every horizon has 8-11 clean
dates against `MIN_FORECAST_DATES = 20`, so by the project's own publication
rule none of this is quotable yet. **And the panel is not maturing.** As of 2026-08-25 the newest
`item_forecasts.created_at` is 2026-08-20 and the newest resolved outcome is
2026-08-17 — nothing has been forecast or scored in five days, because the
forecast chain is part of the billing pause. The "~1 date/day, mature around
2026-09-06" estimate assumed a running chain and is wrong. The real clock starts
when the workflows come back: each horizon needs 20 minus its current count
(h=3: 9, h=7: 9, h=14: 12, h=30: 18) further chain days, so h=3/7 mature ~9 days
after resume and h=30 not for ~18. The direction and consistency of the result are what
this establishes; the magnitudes should be re-read at 20 dates.

## Re-read after the first local chain run (2026-08-25)

The chain was restarted from a laptop (`run_forecast_local.sh`), which resolved
49,830 backlogged outcomes and moved the panel to 41,002 rows / 20 forecast
dates. The result is unchanged on 26% more data:

| h | dates (was) | skill (was) | 90% CI |
|---|---|---|---|
| 3  | 14 (11) | -0.056 (-0.059) | [-0.074, -0.041] |
| 7  | 15 (11) | -0.050 (-0.063) | [-0.080, -0.026] |
| 14 | 10 (8)  | -0.098 (-0.113) | [-0.152, -0.048] |
| 30 | 2 (2)   | +0.048 | [+0.015, +0.078] |

Same signs, CIs still clear of zero, magnitudes drifting slightly toward zero as
the panel fills. h=30 did not move: its forecasts have not matured.

## Where this leaves the decision

Both halves of the GBM have now lost to a featureless null on served data:
the width to climatology (~30% narrower, clean label, every horizon), and the
centre to a random walk (this). Retiring it collapses the serving path to
climatology + last price and takes the retrain, the 28-feature set and the
artifact-flag-matching machinery out of the daily chain.

The one thing that argues for waiting is the date count above. The cheap,
honest sequence is: re-run this at 20 dates per horizon after the panel matures,
and if the sign holds, ship the retirement then. `--gate` answers "has it
matured?" with an exit code, so the re-check is one command and needs no one to
read the table:

    backend/venv/bin/python backend/scripts/centre_vs_lastprice.py \
        --archive-dir ../cs2-oracle-data/price-archive --gate

It exits 1 while any horizon is short and 0 when all four are ready — which
makes it safe to wire into the chain or a scheduled check rather than
remembering to run it.

Because the panel is frozen, the h=3/h=7 legs could also be matured immediately
by RESUMING the chain (they need 9 chain-days each), which is a billing
decision, not a modelling one.

## Why the retirement is not running yet

Actions minutes are exhausted and all nine workflows are `disabled_manually`
until the billing reset. This replay is local-capable and needed neither; the
serving change that follows a confirmed verdict does.
