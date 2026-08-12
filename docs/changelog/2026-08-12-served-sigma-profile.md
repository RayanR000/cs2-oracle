# The tilt is real where the band is served — and the exponent flips it rather than flattening it

**Date:** 2026-08-12
**Runs:** `31643819235` (control) / `31643829741` (arm, `sigma_exponent=true`), commit `0d9f426`,
`horizons=matrix`, `replay_anchors=2026-04-15,2026-05-16,2026-06-16,2026-07-09`
**Instrument:** `_coverage_by_sigma_rows` / `sigma_tilt_pp` in `scripts/replay_serving.py`, new in
`0d9f426`
**Follows:** `2026-08-12-sigma-exponent-paired-read.md` §"Not measured here"
**Reverses:** the refutation of the prod-implied elasticity in
`2026-08-12-sigma-tilt-confirmed-on-oof-residuals.md`
**Status:** ✅ the tilt exists on the serving path — first measurement. ❌ `SIGMA_EXPONENT`
**overcorrects it**, flipping the ramp's sign at 3 of 4 horizons and making it *worse in magnitude*
at 7d and 14d. **The flag stays off, now for a second and better reason.**

## What was unmeasured

Every tilt number on record — the 62→93 / 60→95 / 57→96 / 52→97% decile ramps — is measured on the
**OOF calibration records**. The paired read of 2026-08-12 could therefore say the arm costs 8pp of
marginal coverage but not whether it flattened the profile it exists to flatten. This adds the
served profile, as quintiles of the served half-width.

**The stratifier is exact, not a proxy.** `conformal.band` sets the half-width to
`q_hat * sigma ** beta`, strictly increasing in `sigma` for any `beta > 0`, so its quantiles are
`sigma`'s quantiles and the item ordering is identical under both arms — which is what lets the two
tables be read stratum by stratum. Quintiles rather than deciles because one anchor serves ~1,000
rows.

## The tilt is real at serving, and much milder than on the calibration set

Control, n-weighted over the four anchors. Stratum 1 is the calmest fifth.

| h | q1 | q2 | q3 | q4 | q5 | ramp | marginal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | 70.18 | 74.30 | 78.31 | 80.99 | 85.56 | **+15.66** | 77.86 |
| 7 | 78.83 | 81.93 | 88.28 | 87.44 | 88.50 | **+9.94** | 84.99 |
| 14 | 80.00 | 84.27 | 85.58 | 86.15 | 88.15 | **+8.10** | 84.83 |
| 30 | 82.34 | 88.75 | 88.52 | 84.64 | 91.55 | **+9.13** | 87.16 |

Positive at **15 of 16** anchor-horizon cells. Same sign as the OOF profile and **a third to a half
its size** — the OOF ramp is +31 to +45pp. That gap is itself informative and is the reason the rest
of this entry lands where it does.

## The exponent flips the ramp instead of removing it

| h | ramp control | ramp arm | Δ, paired | t (3 df) | \|ramp\| |
|---|---:|---:|---:|---:|---|
| 3 | +15.66 | **−9.50** | −25.16pp | −6.50 | 15.66 → 9.50 |
| 7 | +9.94 | **−13.57** | −23.51pp | −8.49 | 9.94 → **13.57** |
| 14 | +8.10 | **−14.54** | −22.64pp | −5.52 | 8.10 → **14.54** |
| 30 | +9.13 | **−2.48** | −11.61pp | −3.27 | 9.13 → 2.48 |

Negative at **14 of 16** cells, and the change is significant at 4/4 (critical 3.18 on 3 df). The
calm items now get bands that are too *wide* and the volatile ones bands that are too *narrow* — the
original defect, inverted. **At 7d and 14d the served band is more lopsided after the correction
than before it.** Only 30d is genuinely improved.

Read the **ramp**, not `sigma_tilt_pp`: the ramp is level-free, while the tilt statistic is measured
against a fixed 80% and the arm's marginal level falls ~8pp, which inflates it. This is the same
level-matching trap the research index already warns about, in a place where matching is impossible
because the level is itself under test.

## The exponent is fitted on the wrong population, and that reinstates a refuted read

Interpolating between the two arms, the exponent that flattens the **served** profile is

| h | fitted on OOF (served by the arm) | implied by the served profile | prod-implied read, published as refuted |
|---|---:|---:|---:|
| 3 | 0.4241 | **~0.64** | 0.798 |
| 7 | 0.3672 | **~0.73** | 0.692 |
| 14 | 0.3370 | **~0.76** | 1.034 |
| 30 | 0.3266 | **~0.47** | 1.150 |

⚠️ **Treat the middle column as a bracket, not an estimate.** It is a straight line drawn between two
points of a curve that is not straight, on four anchors. What it establishes is the **direction and
the order of magnitude**: the served-optimal exponent is roughly **double** what is being fitted, and
it sits far closer to the prod-implied read than to the OOF one.

**So the refutation of 0.798 / 0.692 / 1.034 / 1.150 was a basis confusion.** That read was taken on
~20K production rows with `sigma` reconstructed as `half_pct / q_hat`; it was overruled by a read on
OOF residuals and recorded as "refuted at 4/4, hardest where it said fine." Those are two different
populations, and the one that governs what users are served is the production one. The
reconstruction route has its own problems — that criticism stands — but **the verdict does not**, and
`2026-08-12-sigma-tilt-confirmed-on-oof-residuals.md` should be read with that correction.

This is the **second basis confusion in one day** on this feature. The first cost two cancelled
dispatches when the pre-registered width bar turned out to be a calibration-set number being checked
against a served band (`2026-08-12-sigma-exponent-paired-read.md`). Both have the same shape: a
quantity measured on calibration residuals was assumed to transfer to served outcomes. **It does
not, and on this feature it has now been wrong twice in opposite directions.**

## The horizon story inverts as well

The OOF held-out leg says the exponent **works at 3d/7d and fails at 14d/30d** (−78% / −67% against
−22% / −35%), replicated twice. The served profile says it **overcorrects at 3d/7d/14d and is
roughly right at 30d**. Opposite conclusions about which horizons it suits, from the two bases.
Anyone reaching for "it works at the short horizons" now has to say on which population.

## Status and what follows

`SIGMA_EXPONENT` stays **off**. The paired read had it costing 8pp of marginal coverage; this adds
that it fails at its own job on the serving path at 3 of 4 horizons. Two reasons, independent.

The idea is not dead — the tilt it targets is confirmed at serving for the first time here, at 15 of
16 cells. What is wrong is the **fitting population**: fit `beta` on resolved served outcomes rather
than OOF residuals, and the correction is roughly half as strong as the one currently applied. That
is a different instrument from anything built so far, and it needs a forecast-date panel rather than
a CV fold, so cost it before scheduling it.

## Caveats

- ~200 items per stratum per anchor, four anchors, one artifact per arm.
- `2026-07-09` is an outlier in the control at the short horizons (+36.19 / +28.95 ramp against
  +2.83 / +2.75 elsewhere) and pulls the control means up.
- 30d bands whose lower leg was clipped at zero are no longer `q_hat * sigma ** beta` wide and are
  dropped from the strata rather than mis-assigned; 242 of ~990 were clipped on 2026-08-11.
- Nothing here is level-matched, and it cannot be: the marginal level is part of what is being read.
