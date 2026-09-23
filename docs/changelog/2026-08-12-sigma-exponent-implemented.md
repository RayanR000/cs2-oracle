# The `sigma` exponent is implemented, gated off, and it found a bug in the shipped diagnostic

**Date:** 2026-08-12
**Spec:** `docs/specs/2026-08-12-sigma-exponent-design.md`
**Decision:** `changelog/2026-08-12-the-sigma-scale-is-one-exponent-per-horizon.md`
**Flag:** `SIGMA_EXPONENT=1`, off by default. `model-diagnostics.yml` input `sigma_exponent`.
**Tests:** `backend/tests/test_sigma_exponent.py` (17), suite 2052 → **2069 passing**
**Status:** built and gated off. **No dispatch run, nothing promoted, daily path unchanged.**

## What it does

`conformal.calibrate` and `conformal.band` divide by `sigma ** beta` through one new helper,
`conformal.scale`, so the two cannot disagree about the exponent. `beta` is fitted per horizon by
`conformal.fit_beta` on the same OOF records `q_hat` is fitted on, and persisted as `conformal_beta`
in `meta.json` on the line adjacent to `conformal_calibration`.

Verified end-to-end on the real `_calibrate_conformal` path, synthetic residuals with elasticity 0.4
by construction:

| `SIGMA_EXPONENT` | `beta` | `q_hat` | median half-width | marginal | coverage by `sigma` decile |
|---|---:|---:|---:|---:|---|
| `0` | 1.0000 | 833.43 | 58.09% | 80.0% | `41 61 71 78 83 87 90 94 97 99` |
| `1` | 0.3970 | 151.18 | 52.51% | 80.0% | `80 80 80 80 80 80 80 80 80 80` |

**The `q_hat` column is the whole reason this change is arranged the way it is.** It moves **5.5×**
between the arms because `sigma ≈ 0.07` and `sigma ** 0.4 ≈ 0.35`. A `q_hat` from one arm served at
the other's exponent is wrong by that factor — not partially corrected. Hence: one helper applies the
exponent, both keys are written from the same comprehension domain, a missing `conformal_beta`
defaults to **1.0** (which is exactly what every pre-2026-08-12 artifact was calibrated at), and
`band_beta()` is the only accessor so no call site can produce a `KeyError` or push a NaN into a
served half-width.

Marginal coverage is **80.0% in both arms**, which is the property that let this defect live for
months: `calibrate` takes the (1−α) quantile of its own scores whatever the denominator is, so no
marginal-coverage test can see a tilt. There is now a test asserting exactly that.

## A bug in the shipped `elasticity` diagnostic, found by the new tests

`conformal.elasticity` guarded a degenerate fit with `if denom <= 0`. That is insufficient: for a
**constant** `sigma`, `x - x.mean()` is floating-point noise around 1e-16 rather than exact zero, so
the sum of squares is tiny but *positive*, the guard passes, and the slope is the ratio of two
noises. It returned **0.5** — in range, plausible, and `fit_beta` would have clamped nothing and
persisted it into an artifact as a served exponent.

Fixed at the root with `LOG_SIGMA_MIN_SD = 1e-8` on the standard deviation of `log sigma`. Real data
carries ~0.6, so the guard cannot bind on it. This function already ships in the `sigma_tilt` audit,
so the bug was live — though unreachable there, because a real cross-section always has spread.

## Two deliberate deviations from the spec

**1. The per-fold diagnostic stays at `beta = 1.0`, always.** The spec listed it as a fourth site to
convert. Converting it is wrong: `fold_q_hat` exists to test whether `q_hat` falls as a fold's
training set grows, which requires every fold in **one unit** — and the pooled `beta` does not exist
yet when folds run, so the only available exponent is each fold's own. Using it would make
consecutive `fold_q_hat`s incomparable and silently break the published **0.94 / 0.91 / 0.92 / 0.84×**
series and the expanding-window audit built on it.

Instead the fold loop now reports **`fold_beta`**, which is strictly new information and happens to be
the exact quantity the open 14d/30d dispute turns on: whether the exponent drifts between folds
enough to explain the weak single-fold held-out leg (**−26% / −12%**) against the walk-forward read
(**−84% / −73%**). Measured on real OOF residuals, not the model-free panel.

**2. `fit_beta` reports its clamp separately.** `beta_was_clamped` is a second call rather than a
tuple return, so the caller logs a WARNING when the measured elasticity leaves `[0.2, 1.0]` — a range
no window of this archive has produced, so a clamp that binds is a data problem worth surfacing
rather than a tuning outcome to absorb.

## One earlier test was updated, not weakened

`test_the_audit_reaches_meta_json_and_never_the_band` asserted the literal line
`half = float(q_hat) * np.asarray(sigma, dtype=float)`, because when it was written the exponent was
measured and deliberately unimplemented. It now asserts what that line was standing in for: the
**audit** still cannot assign an exponent, `band`'s default is still `BETA_NEUTRAL`, and
`scale(sigma)` returns `sigma` itself. The reason for the change is recorded in the test.

A new guard replaces the coverage the old assertion gave up: `test_every_band_and_calibrate_call_passes_an_exponent`
counts the call sites and greps each one for an exponent, so a fifth site added later fails the suite
instead of silently serving `beta = 1` against a `beta`-era `q_hat`.

## What is NOT done

- **No dispatch.** Every number above is synthetic or offline. The paired run is the next step, and
  the first thing to read is the **median half-width ratio vs control** — predicted
  **0.87 / 0.86 / 0.84 / 0.77×**. A ratio far from that means `beta` reached some call sites and not
  others.
- **`q_hat` must not be differenced across this flag.** The two arms are in different units. Compare
  coverage and width.
- **The marginal over-coverage is not fixed** (87.2/91.8/90.6/89.0% vs 80%). This closes the
  `sigma`-mix channel — 36–68% of it — and on a calm period the corrected band covers **74–77%**, so
  the level is unresolved in both directions and no 80% claim rests on it.
