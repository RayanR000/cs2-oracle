# Design: the conformal band divides by `sigma ** beta`

**Date:** 2026-08-12
**Decides:** one fitted exponent per horizon, no shrinkage, no non-parametric scale
**Evidence:** `changelog/2026-08-12-the-sigma-scale-is-one-exponent-per-horizon.md` (which arm),
`changelog/2026-08-12-sigma-tilt-confirmed-on-oof-residuals.md` (the tilt, on real OOF residuals),
`changelog/2026-08-12-marginal-over-coverage-is-half-the-sigma-mix.md` (why it also helps the level)
**Status:** designed, **not implemented**

## The change in one line

`conformal.calibrate` and `conformal.band` divide by `sigma ** beta` instead of `sigma`, with `beta`
fitted per horizon on the same OOF records `q_hat` is fitted on, and persisted beside it.

## The one invariant that makes this dangerous

**`q_hat` and `beta` are a matched pair and neither is meaningful alone.** `sigma` is a fraction
around 0.07, so `sigma ** 0.4` is about 0.35 — five times larger — and `q_hat` absorbs that entirely.
A band that divides by `sigma ** 1` using a `q_hat` fitted against `sigma ** 0.4` is **not** a
partial fix; it is wrong by a factor of ~5 and would ship a wildly mis-sized band.

Three consequences that the implementation must honour:

1. Every `q_hat` write persists `beta` in the same operation, or neither is written.
2. Loading an artifact without `beta` must default it to **1.0** — the value that reproduces today's
   behaviour exactly — and must never default `beta` while accepting a `beta`-era `q_hat`.
3. No stored `q_hat` may be compared or differenced across a `beta` boundary. This repo has already
   been burned by exactly this class of comparison (`model_version` encoding config, A/B verdicts
   spanning a statistics fix), so the audit line that prints `q_hat` must print `beta` next to it.

## Call sites

`beta` has to reach all four places `sigma` is turned into a width. Missing any one of them produces
the mismatch above rather than a smaller benefit.

| # | `models/forecaster.py` | what it does | why it must change |
|---|---|---|---|
| 1 | `_calibrate_conformal` ~6640 | `conformal.calibrate(resid, sigma)` | fits `q_hat`; where `beta` is fitted and stored |
| 2 | `_calibrate_conformal` ~6656 | `conformal.band(mid, sigma, q_hat)` → `range_pct` | `range_pct` feeds `_calibrate_confidence`'s thresholds |
| 3 | `predict` ~7253 | `conformal.band(...)` on `_sigma_for_rows(latest_rows)` | **the served band** |
| 4 | CV ~6483 / ~7682 | fold-level calibrate on `fold_sigma` | `fold_q_hat` and the expanding-window audit |

`_sigma_for_rows` itself does **not** change: `beta` applies to the width, not to the clip. The clip
bounds stay in the raw `sigma` space so `sigma_clip` keeps its meaning across the boundary.

## Signature

Extend both functions with a default that is today's behaviour:

```python
def calibrate(residuals_pct, sigma, alpha=ALPHA, beta: float = 1.0) -> float
def band(mid_pct, sigma, q_hat: float, beta: float = 1.0)
```

`beta = 1.0` must be a genuine no-op, not `sigma ** 1.0` recomputed — assert bit-identical output at
`beta = 1.0` so the gate-off path is provably unchanged.

`conformal.fit_beta(residuals_pct, sigma)` wraps the existing `conformal.elasticity` with the
guards the fit needs in production: return **1.0** (not NaN) when fewer than `MIN_CALIBRATION_ROWS`
rows survive, when `sigma` has no spread, or when the result is non-finite. Clamp to `[0.2, 1.0]`:
0.2 is below every value ever measured (0.209 is the minimum across 24 time blocks) and 1.0 means
the fix cannot *widen* the band relative to today by moving the exponent the wrong way. A clamp that
binds is a bug, so log at WARNING when it does.

## Persistence

`meta.json` gains one field, written in the same block as `conformal_calibration`:

```json
"conformal_beta": {"3": 0.4291, "7": 0.4498, "14": 0.4287, "30": 0.3641}
```

Loading (`~8548`) mirrors `conformal_calibration`'s strictness with one deliberate exception: use
`meta.get("conformal_beta", {})` and default a missing horizon to **1.0**, because every artifact
written before this change lacks the key and must keep serving the band it was calibrated for.
`conformal_calibration` stays strict.

## Gate

`SIGMA_EXPONENT=1` off by default, following `LABEL_SMOOTHED_ANCHOR`'s pattern, plus a
`sigma_exponent` input on `model-diagnostics.yml` so the arm can be dispatched paired against a
control on the same commit and the same folds. When the flag is off, `fit_beta` is not called and
`conformal_beta` is written as all-1.0 rather than omitted — an explicit 1.0 is auditable and a
missing key is ambiguous.

## Tests

1. `beta = 1.0` reproduces the current `q_hat` and the current band bit-identically.
2. On synthetic residuals with elasticity 0.4 by construction, `fit_beta` recovers 0.4 ± 0.02, and
   the `beta`-calibrated band's coverage across `sigma` deciles is flat within 10pp where the
   `sigma ** 1` band ramps by 30pp+. (`test_sigma_normalization_fails_when_the_elasticity_is_not_one`
   already pins the defect; this pins the fix.)
3. **Marginal coverage is unchanged** by the exponent on the calibration set — it is 80% by
   construction either way. This is the test that documents why nothing caught the tilt for months.
4. `fit_beta` returns exactly 1.0 on each guard path (too few rows, no spread, non-finite).
5. An artifact with no `conformal_beta` loads with `beta = 1.0` at every horizon.
6. A `beta`-era `q_hat` loaded with a defaulted `beta = 1.0` must be **impossible**: assert the
   writer emits both keys together.
7. The four call sites all use the artifact's `beta` — assert by source inspection the way
   `test_sigma_tilt_audit.py` already asserts `band` divides by `sigma ** 1`, so a fifth call site
   added later fails the suite instead of silently serving `beta = 1`.

## Dispatch, and what it must report

One paired `model-diagnostics.yml` run, arm vs control, same commit, all four horizons. The arm must
report, per horizon: `beta`, `q_hat` beside it, raw and level-matched coverage by `sigma` decile,
marginal coverage, and **median served half-width relative to control** — the offline read predicts
**0.87 / 0.86 / 0.84 / 0.77×**, and a width ratio far from that means `beta` reached some call sites
and not others, which is the failure mode this whole spec is arranged around.

It also settles the open disagreement: whether the tilt fix reaches 14d/30d (**−84% / −73%** on the
walk-forward panel) or barely does (**−26% / −12%** on the single held-out CV fold).

## What this does NOT fix, and must not be claimed to

**The marginal level.** Held out the band still lands 3.9–5.1pp off 80%, and on the *earlier* period
the exponent arms cover **74–77%** — production is closer to nominal there. The exponent removes the
tilt and the `sigma`-mix channel of the level defect (36–68% of it); the residual-law term is
untouched and is the next measurement. **No 80% claim should be attached to the served band on the
strength of this change.** The `confidence` tag is already withdrawn, which is what keeps that
honest.

**Per-date coverage.** 58.2–99.2% per (horizon, date) in production; the date spread barely moves
here (8.0 → 7.3pp at 3d, 5.5 → 6.0pp at 30d — it gets slightly *worse* at 30d).

## Deliberately deferred

A non-parametric scale **at 30d only**, where the binned arm cuts the residual tilt 2.44 → 1.47pp.
Worth revisiting after the exponent lands and only if 30d's residual tilt is still the largest thing
left; it costs twenty persisted bin edges and their medians against one float.

## Cost

~40 lines across `conformal.py` and `forecaster.py`, ~120 lines of tests, one paired dispatch
(~15 min each arm, inside the 30-minute wall-clock cap if the arms run sequentially).
