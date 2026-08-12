# The learned band scale is built and measured: cheap, better than the exponent, still not shippable

**Date:** 2026-08-12
**Runs:** `31649561391` (control) / `31649571169` (arm, `learned_scale=true`), commit `5b39760`,
`horizons=matrix`, `replay_anchors=2026-04-15,2026-05-16,2026-06-16,2026-07-09`
**Implements:** the cheap alternative to CQR costed in this entry's §"Why not quantile regression"
**Follows:** `2026-08-12-served-sigma-profile.md`
**Status:** ✅ built, gated off as `LEARNED_SCALE=1`, and it costs **26.6s**. ⚠️ It beats
`SIGMA_EXPONENT` at 7d/14d and is the only arm that flattens 30d — but it **overcorrects at 3d/7d**
on the ordinary anchors and loses marginal coverage at 3d. **Stays off.**

## Why not quantile regression

The obvious redesign — train low/median/high models and conformalize the interval (CQR) — was
costed from the timings of `31643819235` and **does not fit**. The out-of-fold conformal CV is
**646.8s of 1017.8s (63.5%)** of training, and CQR needs p10/p90 *out of fold*, so that phase fits
three boosters per fold instead of one: **2454s (40.9 min)** with per-fold overhead shared,
**2778s (46.3 min)** at a straight 3x, against a **30-minute cap for the whole job**. It is also not
a new idea here — 24 p10/p90 boosters were deleted on 2026-08-04 at 223.2s of a 381.2s budget and
39-48% coverage (`2026-08-04-minimal-model-results.md`). That coverage figure is *not* an argument
against CQR, which repairs exactly that failure; the clock is.

What fits is a scale model: one small booster per fold predicting `log |residual|` from the item's
features, on records that already exist. Split conformal still builds the interval, so the
distribution-free guarantee is untouched.

**Measured cost: 26.6s across four horizons** (7.8 / 6.3 / 6.3 / 6.2s for 8-9 cross-fit models plus
a serving model on 155-176K rows each), i.e. **2.6%** of training against a predicted ~40s. The
affordability argument holds.

## The result

Mean over four anchors. `ramp` is q5 − q1 of served coverage across quintiles of the served width;
`tilt` is mean |coverage − 80| across the strata.

| h | \|ramp\| control | \|ramp\| **arm** | \|ramp\| `SIGMA_EXPONENT` | tilt | \|marginal − 80\| | width |
|---|---:|---:|---:|---|---|---:|
| 3 | 15.66 | **10.02** | 9.50 | 4.75 → **7.93** | 2.14 → **7.93** | 0.881x |
| 7 | 9.94 | **11.41** | 13.57 | 5.46 → 3.43 | 4.99 → **1.27** | 0.815x |
| 14 | 8.10 | **0.65** | 14.54 | 4.83 → 3.83 | 4.83 → 3.43 | 0.818x |
| 30 | 9.13 | **7.19** | 2.48 | 7.16 → 5.73 | 7.16 → 5.73 | 0.776x |

It beats the exponent at 7d and 14d, loses to it at 30d, and narrows the band 0.78-0.88x throughout.

## ⚠️ The 14d "flat profile" is cancellation, not flatness

**Read this before quoting the 0.65.** Per anchor, the 14d ramp is **−4.09 / −15.74 / −8.95 /
+31.39**. `2026-07-09` is a systematic outlier: the arm's ramp there is strongly **positive at 4 of
4 horizons** (+28.00 / +24.12 / +31.39 / +26.51) while negative at the other three anchors. Drop it:

| h | ramp, 3 ordinary anchors |
|---|---|
| 3 | +8.81 → **−22.69** |
| 7 | +3.60 → **−23.26** |
| 14 | +9.86 → −9.59 |
| 30 | +11.55 → **+0.75** |

So on ordinary dates the learned scale **overcorrects harder than the exponent did** at 3d and 7d,
and 30d is the genuine success. The four-anchor mean at 14d would have been published as a clean win;
it is two large opposite errors averaging to zero.

**Four horizons all reversing on one date is a property of that date, not of the model**, and it is
the thread worth pulling next.

## An instrument correction

`_coverage_by_sigma_rows`' docstring claimed the served half-width's quantiles are `sigma`'s
quantiles, so the strata describe the same items in both arms. That is true for `SIGMA_EXPONENT`,
which is a monotone transform of `sigma`. **It is false for `LEARNED_SCALE`**, which is a different
variable and reorders items freely — its stratum 3 and the control's stratum 3 are different cohorts.
The table still answers the calibration question (*is coverage flat across the widths served*), and
the `ramp` survives the reordering, but a stratum-by-stratum diff against a `sigma` control is not
like-for-like. Docstring fixed in this commit; no number above depends on the wrong reading.

## The pattern, now at three for three

Every scale tried is calibrated to **exactly 80%** on its own records and lands somewhere else when
served:

| scale | served marginal, 3/7/14/30d |
|---|---|
| `sigma` | 77.9 / 85.0 / 84.8 / 87.2 — **over**-covers |
| `sigma ** beta` | 69.7 / 76.7 / 76.9 / 84.9 — **under**-covers |
| learned | 72.1 / 78.7 / 76.6 / 85.7 — **under**-covers |

Three different answers to "which variable sets the width", three different displacements between
calibration and serving. **That points away from the width variable being the lever at all** and
toward the calibration population not matching the served one — the same finding that cost two
cancelled dispatches and reversed a published elasticity earlier today. A fourth scale variable is
not the next experiment.

## What is on `main` and what it is for

`LEARNED_SCALE=1`, off by default, with the machinery documented at its definitions:

- `q_hat` is calibrated against a **cross-fitted** scale (every row scored by a model that never saw
  its fold). A scale fitted on the residuals it normalises matches them better than it will match a
  served item's, which makes `q_hat` too small and the band under-cover in production while looking
  flawless offline.
- The target is `log |residual|` under L1 loss, so only **relative** variation matters — `q_hat`
  absorbs any constant factor — and an item's band is not set by its worst historical day.
- `sigma` is an input, so the learned scale **nests** the current band: it could have reproduced
  `s ~= sigma`. It did not, which is why this is a real result rather than an inconclusive one.
- `q_hat`, the booster, its norm, its clip and its column list are a matched set, dropped together
  when any is missing. A `q_hat` in the learned scale's units served against `sigma` is an unrelated
  band, not a degraded one.
- `LEARNED_SCALE` and `SIGMA_EXPONENT` raise if both are set. They are alternatives; applying both
  re-tilts the band the other way.

Verified end to end on a synthetic cohort built so `sigma` is the wrong variable — the `sigma` band
ramps 45.3 → 98.9% across quintiles (+53.6pp) while the learned band holds 80.5 → 80.0% (−0.6pp) at
the same 80% marginal. **That is a constructed case and it is not what the archive did**, which is
the gap between this entry's §"The result" and the machinery working correctly.
