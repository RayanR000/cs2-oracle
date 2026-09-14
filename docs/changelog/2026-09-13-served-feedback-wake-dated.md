# Served-feedback q_hat relevelling: verified dormant-correct, wake is dated

**Date:** 2026-09-13. No code change — verification + countdown only.
**Design:** `docs/superpowers/specs/2026-08-16-served-outcome-feedback-calibration-design.md`.
**Mechanism:** `models/served_recalibration.py`, wired `train()` → `meta.json`
→ `served_qhat_multiplier()` at predict (`forecaster.py:6199,8364,10045`).

## Wake status: dormant, and correctly so

The geometry floor (`SHRINK_K_SERVING_START = 2026-09-06`, the latest of three
cutovers) restarts the date count, and only 7 days have passed since it:

| h | post-floor resolved dates (gate: 20) | status |
|---|---|---|
| 3 | 3 (16,608 rows) | dormant |
| 7 | 0 (maturity lag) | dormant |
| 14 | 0 (maturity lag) | dormant |
| 30 | 0 | dormant |

Fitting below the gate would repeat the exact error class closed this week
(thin-bin quantiles under regime shift). The gate holds; nothing is forced.

## The lever has a live target

h=3 current-geometry (K=320) served coverage over the 3 post-floor dates is
**92.4% against the 80% nominal** (+12.4pp) — the marginal over-coverage the
feedback exists to remove persists on the serving geometry. Diagnostic-only
factor with the gate overridden to 3 dates points the expected direction
(narrow, raw P80 below the 0.5 clamp → reported 0.5 with clamp WARNING): the
sign is consistent with the coverage read, the magnitude is unquotable on 3
dates (~3 independent episodes), which is why it stays a diagnostic.

## Countdown (chain is serving daily at all horizons, 5,536 rows/date)

At the current pace the gate crosses ~09-29 (h=3), ~10-03 (h=7), ~10-10
(h=14), November (h=30), each resolving +h days after its 20th post-floor
forecast date. Activation is automatic at the next **full** retrain after each
crossing (`train()` computes, `meta.json` persists, predict applies) — but
predict-only runs do not recompute, so each crossing needs a
`mode=full`/`ALLOW_DRIFT_RETRAIN=1` run to actually wake. If the chain sits in
predict-only through a crossing, the wake lags silently (the factor stays
absent = 1.0, by design no alarm).

## Checks run

- 31 served-recalibration tests green (estimator, wiring, panel-schema).
- Post-floor panel read live against prod (read-only): counts above.
- No threshold, clamp, or floor constant touched.

Re-check: re-run the panel count per horizon against `forecast_outcomes`
with `forecast_date >= '2026-09-06'`; wake is real when a horizon shows 20.
