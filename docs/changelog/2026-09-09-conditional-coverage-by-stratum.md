# Conditional coverage by stratum: tier and staleness are flat, width is the known tilt

**Date:** 2026-09-09
**Script:** `backend/scripts/conditional_coverage_by_stratum.py`
(+ `backend/tests/test_conditional_coverage_by_stratum.py`, 9 tests)
**Status:** diagnostic only — no modelling or serving change.

`measure_conditional_qhat.py` refuted DATE-conditional q_hat on 2026-08-12.
Per-tier/liquidity/family q_hat is a different axis and was untested, so this
reads coverage per stratum on the served panel before anyone builds the arm.

## Panel

Data-repo `forecast_outcomes.parquet`, served cohort (>=$1),
`excluded_forecast_date` applied, anchor-wedge rebased exactly as
`centre_vs_lastprice.py` (r_hat off the model's own quote, half-widths
transplanted as offsets): 42,864 rows / 20 dates (h=3: 14, h=7: 16, h=14: 11,
h=30: 2). Every horizon is below `MIN_FORECAST_DATES = 20`, so directions are
informative and magnitudes are not quotable. h=30 (2 dates, no CIs) is reported
and ignored throughout.

## Verdict per axis

**Tier — FLAT where the rows are, skip the arm.** Tiers 1–3 (99% of rows) sit
within ~2pp of each other at h=3/7/14 (h=3: 90.2/92.5/92.6%; h=7:
90.6/94.2/94.1%; h=14: 84.1/83.8/80.7%). The max-min spread (5.4/7.0/11.2pp)
is carried by tier-4 (>$100, n = 86/128/98, CI ±5–8pp) — a thin tail, not a
stratum to fit a separate q_hat on.

**Staleness — FLAT, skip.** fresh vs stale: h=3 90.3 vs 93.4% (3.2pp),
h=7 90.3 vs 90.7%. The wider spreads at h=7/14 come from the `unknown`
(pre-2026-08-08 NULL) bucket, which is a provenance label, not a liquidity
signal.

**Family — FLAT for the bulk, one footnote.** pistol/rifle/smg agree within
~1–2pp at every horizon. `musickit` under-covers slightly at h=3 (84.7%,
n = 484, CI [81.4, 87.4] vs pistol [88.7, 91.3]) — worth a re-read at 20
dates, not an arm today. Everything else that deviates is n ≤ 110.

**Width — SPREAD, but this is the known sigma tilt, not a new finding.**
Narrow vs wide tertile: h=3 86.3 vs 95.6%, h=7 84.3 vs 95.9%, h=14 77.0 vs
86.1%, CIs non-overlapping. Low-sigma under-covered, high-sigma over-covered
is `2026-08-12-the-band-is-tilted-in-sigma.md`, confirmed on OOF residuals.

## The width-axis Mondrian already exists — as dead code

`models/conformal.py` carries a complete Width-Adaptive implementation
(`calibrate_signed_waci`, `waci_lookup`, `band_signed_waci`) and NOTHING
calls it: no forecaster wiring, no flag, no script, no test, no changelog.
The diagnostic routes the width finding there, not to a new build: the next
step on this axis is to MEASURE WACI (offline sigma-strata read in the style
of `measure_conditional_qhat.py`'s level-matched columns, then a served
confirm), and the per-tier/per-family arm stays skipped per the table above.

## Reproduce

```
venv/bin/python scripts/conditional_coverage_by_stratum.py \
    --archive-dir ../../cs2-oracle-data/price-archive
```
