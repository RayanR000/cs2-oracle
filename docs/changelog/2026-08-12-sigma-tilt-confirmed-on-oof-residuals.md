# The `sigma` tilt is confirmed on real OOF residuals, and it decays with the horizon

> ## ⚠️ The lead in §"One lead it does open" is SIZED, and its 30d objection is REFUTED.
> `changelog/2026-08-12-marginal-over-coverage-is-half-the-sigma-mix.md`. The served `sigma`-mix
> shift buys **+4.0 to +4.9pp** of the marginal excess — **36–68%** of it — with the `β = 1` placebo
> at **0.0000pp**, so the channel is the tilt and nothing else. The **0.93×** at 30d quoted below is
> an artifact: `sigma` carries **no horizon term**, so that ratio rested on production's single 30d
> forecast date via the implied `half_pct / q_hat` route. Measured directly it is **1.28–1.29× at
> all four horizons**, and 30d moves from `A = −0.22` to **`A = +0.51`**. The caveat "it predicts
> the opposite at 30d" is withdrawn. ⚠️ 32–64% of the excess remains, and by identity it is now
> **named**: a shift in the residual law at given `sigma`, not a seventh unrelated cause.
> Consequence for §"Next": normalising by `sigma ** β` also makes marginal coverage invariant to the
> served mix, which is a second and independent argument for the exponent **at all four horizons**,
> not only the two where the conditional fix held out.

**Date:** 2026-08-12
**Run:** `31619383780` (control, `cv_diagnostic_classifier=q50`, all four horizons, commit `58c6cf9`)
**Confirms:** `2026-08-12-the-band-is-tilted-in-sigma.md`
**Refutes:** the elasticity figures in `2026-08-12-conformal-basis-follows-serving.md`
**Status:** ✅ the tilt is real and the offline instrument was accurate. ⚠️ the *remedy* transfers
out-of-fold at 3d/7d and only partially at 14d/30d. **Nothing shipped.**

## What was being settled

The offline read found `d log|residual| / d log sigma` at **0.408 / 0.401 / 0.363 / 0.327** on a
model-free proxy (`r̂ = 0`). The read already on record found **0.798 / 0.692 / 1.034 / 1.150** from
~20K prod outcomes with `sigma` reconstructed as `half_pct / q_hat`, and called the tilt
second-order — *"at h=14 and h=30 it is fine, or slightly under-corrected."* Same sign at the short
horizons, opposite verdict at the long ones, and a 3× disagreement on magnitude. This run measures
it on the **real OOF conformal records** — the population `q_hat` is actually fitted on.

## The elasticity

| h | offline, model-free | **real OOF** | excl. clipped rows | held-out fit | prior prod read |
|---|---:|---:|---:|---:|---:|
| 3 | 0.408 | **0.429** | 0.428 | 0.425 | 0.798 |
| 7 | 0.401 | **0.369** | 0.353 | 0.354 | 0.692 |
| 14 | 0.363 | **0.350** | 0.333 | 0.315 | 1.034 |
| 30 | 0.327 | **0.313** | 0.281 | 0.276 | 1.150 |

`n` = 175,739 / 157,805 / 157,338 / 155,617 over 9 / 8 / 8 / 8 folds.

**The model-free instrument was accurate to 0.014–0.032 at every horizon**, which retires the
biggest declared confound in the offline read and makes that instrument reusable.

**The prior figures are refuted at 4 of 4**, and most consequentially at the two horizons they
declared fine: 14d and 30d are where the tilt is *worst*. Their `sigma` was implied from a served
half-width divided by a `q_hat` that differed by up to ~3% across the panel, on ~20K rows; this is
~157K rows with `sigma` read directly off the record it was calibrated with. The mechanism of that
error is visible in their own table — `p80(s)` below 1 in 19 of 20 strata is the *uniform*
over-coverage, and pooling it with the tilt is exactly what level-matching separates.

**It is not the clip.** 1.80 / 1.88 / 1.88 / 1.88% of rows sit at the `sigma` floor or cap, and
excluding every one of them moves the elasticity *further from* 1.0, not toward it.

## Coverage by `sigma` decile, level-matched to 80%

Marginal coverage is forced to exactly 80% in both columns, so only the spread is comparable.

| h | `β = 1` — what production serves | err | at fitted `β` | err |
|---|---|---:|---|---:|
| 3 | 62 69 72 76 80 83 87 88 90 **93** | 7.97pp | 81 80 80 80 80 80 82 80 80 77 | **0.73pp** |
| 7 | 60 67 72 75 79 83 88 89 90 **95** | 9.11pp | 81 80 80 80 79 80 82 81 80 76 | **0.95pp** |
| 14 | 57 66 73 77 79 83 88 89 91 **96** | 9.49pp | 79 80 81 81 79 80 82 81 81 77 | **1.09pp** |
| 30 | 52 63 74 81 83 83 85 88 92 **97** | 10.07pp | 73 79 83 85 83 80 79 78 78 81 | **2.56pp** |

A **31–45pp ramp**, monotone, at every horizon. The quietest tenth of the cohort is covered
**52–62%** against a stated 80%, and the most volatile tenth **93–97%**.

## The held-out leg, which is the one that can fail

`β` fitted on every fold but the last and scored on the last:

| h | held-out fold | err at `β = 1` | err at fitted `β` | reduction |
|---|---:|---:|---:|---:|
| 3 | 8 | 7.26pp | **1.78pp** | −76% |
| 7 | 7 | 7.05pp | **2.65pp** | −62% |
| 14 | 7 | 6.01pp | **4.44pp** | −26% |
| 30 | 7 | 6.09pp | **5.39pp** | −12% |

**This is the finding the pooled table above cannot give.** The tilt is confirmed at 4/4, but a
single global exponent only *fixes* it at the short horizons. At 30d it recovers 0.70pp of 6.09pp,
and the pooled corrected profile there is visibly not flat — `73 79 83 85 83 80 79 78 78 81`
overshoots the quiet end while still under-covering it. Two reasons, both consistent with the rest
of this repo: the exponent drifts across folds (30d fits 0.313 pooled against 0.276 on the earlier
folds), and a 30d fold holds far fewer effective observations because its windows overlap.

So the honest scope of the remedy is **3d and 7d**, where it removes three quarters and two thirds
of the conditional error out of fold. 14d and 30d need either a per-fold/shrunk exponent or a
non-parametric scale — and that is a second decision, not this one.

## What this does NOT fix

**The marginal over-coverage.** Production covers **87.2 / 91.8 / 90.6 / 89.0%** against 80%, and
level-matching removes exactly that quantity before any of the above is measured. On the
calibration set marginal coverage is 80% by construction, so the exponent is orthogonal to the
headline number. Six causes have now been examined and the marginal defect belongs to none of them.

**One lead it does open, stated as a lead.** With `β ≠ 1`, a shift in the *served* `sigma`
distribution relative to the calibration one moves marginal coverage — and served `sigma` is
**1.28 / 1.34 / 1.33 / 0.93×** the calibration median (`2026-08-12-conformal-basis-follows-serving.md`).
High-`sigma` rows are the over-covered ones under `β = 1`, so a served distribution shifted *up*
predicts marginal over-coverage at 3d/7d/14d. ⚠️ **It predicts the opposite at 30d**, where served
`sigma` is 0.93× and the band still over-covers at 89.0%. So it is at best a partial mechanism and
it needs its own read.

## What did not change

`q_hat` = **95.86 / 142.96 / 207.41 / 315.09** and every `fold_q_hat` in the
`Expanding-window audit` line reproduce run `31611508808` exactly (`127.0 → 61.5 → 91.2 → …` at
h=3). The fold tag added for the held-out leg is ignored by `_calibrate_conformal`, the audit runs
after calibration, and `conformal.band` still divides by `sigma ** 1`. The daily path is untouched
and no artifact was promoted.

## Next

Implementing the exponent is now justified **at 3d and 7d** on a held-out read. It needs:

1. `beta` fitted per horizon in `_calibrate_conformal`, persisted beside `sigma_clip` in
   `meta.json`, and applied in **both** `calibrate` and the serving `band` — a served band built on
   `sigma ** 1` from a `q_hat` fitted on `sigma ** beta` is worse than either alone.
2. A decision for 14d/30d that is *not* the same constant: shrink `beta` toward 1 by its
   fold-to-fold variance, or drop the parametric form. Do not ship one exponent across four
   horizons on the strength of the 3d/7d result.
3. `LABEL_SMOOTHED_ANCHOR`-style gating and a paired dispatch, since this moves every served
   half-width.

The offline instrument (`backend/scripts/measure_conditional_qhat.py`) is now validated against
real residuals to 0.03, so 1 and 2 can be designed offline before any dispatch is spent.
