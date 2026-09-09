# WACI halves the sigma tilt — but the placebo passes too, so the bar is void

**Date:** 2026-09-09
**Script:** `backend/scripts/measure_waci.py`
(+ `backend/tests/test_measure_waci.py`, 8 tests)
**Status:** diagnostic only — no wiring, no serving change. WACI stays dead code.

`conformal.py`'s Width-Adaptive arm (`calibrate_signed_waci` etc.) had zero
callers; the 2026-09-09 stratum diagnostic said width is the one axis with
real spread. This measured it: pooled signed pair vs per-width-bin signed
pairs vs shuffled-scale placebo, expanding history under the H+13 embargo,
model-free realised-return residuals, 1,176–1,272 test dates per horizon.

## Numbers (level-matched sigma-strata err, pp; lower is better)

| h | S0 pooled | WACI | placebo (shuffled) |
|---|---|---|---|
| 3 | 8.52 | **4.52** | 6.32 |
| 7 | 8.74 | **4.54** | 6.72 |
| 14 | 9.01 | **4.52** | 6.71 |
| 30 | 9.15 | **4.85** | 6.78 |

WACI is better than pooled at 4/4 with raw marginal in band at 4/4 — and the
placebo is better than pooled at 4/4 too. **Per the pre-registered placebo
clause the bar is VOID, not passed.** The metric rewards something besides
scale information.

## What the placebo is telling us

Roughly half the gain (8.7→6.7 of 8.7→4.6) comes from the binning machinery
alone: ten quantile fits on random subsets, kernel-averaged at serve time.
That is bagging/shrinkage of a noisy tail quantile, not conditioning — and it
trades: both WACI and placebo UNDER-cover the top sigma decile (WACI 66-68%,
placebo 52-54%, vs pooled 95-98%). Mean-abs-err does not penalise the
direction of the miss, so trading top-decile over-coverage for
under-coverage scores as a fix. A ship that under-covers the most volatile
decile by 14-28pp is not a calibration improvement.

The other half (placebo 6.7 → WACI 4.6, same direction at all four horizons)
is consistent with real scale information — but it was not the registered
contrast, so it is a hypothesis for a follow-up, not a verdict.

## Specificity (good)

The date-dimension error barely moves (S0 8.4-11.4pp → WACI 8.1-10.6pp):
WACI fixes the sigma dimension specifically, as designed. Not a general
variance reduction wearing a Mondrian costume.

## Next, if anyone picks this up

1. Re-register with **treatment-vs-placebo** as the primary contrast, plus a
   guardrail the current bar lacked: no decile below 70% (kills the
   top-decile trade silently passing mean-abs-err).
2. Test **bagged-pooled** (mean of K bootstrap quantile pairs, served as one
   pair) as a second control — if it matches the placebo, the machinery half
   of the gain ships for free with no per-item lookup and no top-decile
   collapse.
3. Only then: wire WACI behind a flag and confirm on real OOF residuals, then
   served dates. The served panel has <20 dates/horizon; nothing here is
   quotable against it yet.

## Reproduce

```
venv/bin/python -m scripts.measure_waci --horizons 3,7,14,30 --out /tmp/waci_measure.csv
```
