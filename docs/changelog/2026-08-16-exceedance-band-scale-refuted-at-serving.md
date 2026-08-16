# The exceedance band scale is refuted at serving — do not ship

**Date:** 2026-08-16. Paired A/B, `EXCEEDANCE_SCALE` off by default and STAYS off.
**Runs:** control `31961733400` vs arm `31961738204`, both on `exceedance-band-scale` at one commit,
Model Diagnostics, anchors `2026-04-28,2026-05-19` (audited-clean, all four horizons, one collection
regime), HP reused (cached), `--train-only` + served replay on the ≥$1 cohort.
**Builds on:** `2026-08-16-exceedance-band-scale-built-gated-off.md` (the flag), and the offline
positive `2026-08-16-exceedance-probability-improves-band-conditional-coverage.md`.

## Result

Arm vs control, marginal `BAND COVERAGE` and marginal half-width, ≥$1, per anchor:

| h | cov Δ (arm−ctrl), pp | half-width ratio |
|---|---|---|
| 3 | +0.4 / +0.4 | 0.99 / 1.03 |
| 7 | +0.9 / +0.1 | 0.97 / 1.02 |
| 14 | +0.9 / +0.9 | 0.99 / 1.02 |
| 30 | **−5.4 / −6.6** | **0.82 / 0.79** |

The arm **did** apply (top-sigma-stratum half-width e.g. h=3 @04-28: 26.21 → 52.31; `meta.json`
carries `exceedance_scale`). It is a no-op at 3/7/14 on both marginal coverage and width, and at 30d it
narrows ~20% and pulls the marginal over-coverage (88.3/86.9%) down toward nominal (82.9/80.3%).

## Why it fails — the axis mismatch

The offline win was on the **realised-|resid|** quintile (the items that actually moved). Production
stratifies and is judged on the **sigma / served-half-width** axis, and the two disagree — the
trailing-vs-forward volatility gap this project has hit repeatedly. On the served axis:

- The **under-covered** cells are the **LOW-sigma** items (e.g. h=30 @05-19 stratum 1/5 covers 83.8%),
  not the high-sigma ones, which **over-cover** (stratum 5/5 ~90-98%).
- `p_exceed` rises with sigma, so `sqrt(p)` **widens high-sigma bands and narrows low-sigma bands** —
  it *amplifies* the existing tilt rather than flattening it. It doubles width on the already-
  over-covering top stratum (pure waste) and narrows the already-weakest low stratum.

The damage concentrates at h=30, where narrowing the low-sigma stratum **collapses its coverage
83.8% → 66.2%** (@05-19; 80.8% → 71.8% @04-28). At short horizons the reallocation is small enough to
wash out marginally.

## Verdict

Fails the pre-registered ship criterion (hold marginal coverage AND improve conditional tail coverage
at ≤~15% width). It is the **fourth band-width scale** measured, with the same OOF-positive /
serving-neutral-or-negative signature as `sigma_exponent` and `learned_scale`
(`.claude/rules/training-budget.md`: "The width variable is not the lever. Do not propose a fourth
one." — now measured, not assumed). The offline `sigma × sqrt(p)` result does not survive the
transfer from the |resid| axis to the served sigma axis; **do not carry the offline coverage numbers
across that boundary**, the same rule that caught `sigma_exponent` twice.

Caveats: two anchors, ~1,050 ≥$1 rows per cell; the by-sigma strata are arm-defined so cross-arm
stratum membership differs — but the marginal numbers and the mechanism are consistent across both
anchors and all four horizons.

## Disposition

The code stays **built and gated off**, like its two refuted siblings, so the instrument and the arm
survive for re-measurement rather than being deleted. `EXCEEDANCE_SCALE=0` (default) serves the
plain-sigma band byte-for-byte. No served headline, and the `MIN_FORECAST_DATES=20` bind is moot — the
arm is refuted before it reaches that gate. The exceedance **signal** itself is unaffected (still the
one market-orthogonal magnitude signal); only its use as a *served* band-width scale is refuted.
</content>
