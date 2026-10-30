# Pre-registration: reopening served direction (`DIRECTION_DISCLOSED`)

**Date:** 2026-10-08, written **before any in-window number is computed.**
**Status:** FROZEN when the PR that adds this file merges. Nothing in the window may be
scored before that merge, and nothing below may change after it except under a dated
*Amendments* entry written before the read.

**Follows:** `changelog/2026-09-10-served-direction-withheld.md`, which withdrew the served
call (PT negative at every horizon) and set the reopen rule: positive PT at 20 or more
dates. This doc replaces that rule with a stricter one and fixes the window, because the
old rule has two gaps (below).

## Seen before freezing (disclosed)

A read-only probe on 2026-10-08 (prod `forecast_outcomes`, ≥$1, scored through
`scripts/backtest_accuracy.py::_records_from_frozen_outcomes`, the backtest's own derivation).
The latest forecast date it saw was **2026-10-04**.

| Window | h | dates | DA | realised down | PT excess | t | p |
|---|---|---|---|---|---|---|---|
| all dates | 3 | 44 | 41.3% | 52.8% | −0.58pp | −1.08 | 0.288 |
| all dates | 7 | 46 | 50.3% | 59.9% | −0.03pp | −0.08 | 0.938 |
| all dates | 14 | 40 | 53.1% | 67.5% | +0.06pp | +0.09 | 0.931 |
| all dates | 30 | 22 | 53.2% | 73.5% | −0.13pp | −0.24 | 0.813 |
| ≥ 2026-09-22 | 3 | 13 | 36.0% | 47.9% | +1.41pp | +2.67 | 0.020 |
| ≥ 2026-09-22 | 7 | 9 | 44.9% | 55.2% | +0.88pp | +2.20 | 0.059 |
| ≥ 2026-09-22 | 14 | 2 | 49.5% | 67.2% | +1.76pp | n/a | n/a |

The ≥ 2026-09-22 slice was chosen from the 09-20 retrain (served rank IC turned positive on
09-22, `research/2026-09-28-next-steps.md` item 5), not from these numbers. It is still
one of eight horizon-by-window cells that were looked at. **Every forecast date through
2026-10-04 is therefore exploration, and none of it counts toward this test.** At h=3 after
the retrain, the model called "down" on 39.2% of rows against an actual down rate of 47.9%.

## The two gaps in the old rule

1. **PT positive is not "the call beats the runnable baseline".** PT tests whether calls
   carry information against their *own* marginals. The h=3 slice above has positive PT and
   a DA of 36.0%, **11.9pp below always-down**. Flipping `DIRECTION_DISCLOSED` on PT alone
   would publish a call that loses to a free baseline. Invariant 4 (`backend/AGENTS.md`)
   already says DA is quotable only beside `realised_down_rate`.
2. **Nothing fixed the window or the hurdle.** A slice picked after the fact with p = 0.02
   among eight cells is what the house hurdle (`PT_T_HURDLE = 3.0`, Harvey–Liu–Zhu) exists
   to stop.

## Window and population

- **Window:** forecast dates **≥ 2026-10-05**, a boundary after every date the probe saw, and
  the first date served with the 10-05 refit factors (next-steps item 8) on the
  composition-break-calendar retrain. The weekly retrains on 10-12, 10-19 and later happen inside the window. The
  object under test is the **served system**, not one artifact.
- **Cohort:** `forecast_outcomes`, ≥$1, scored exactly as the backtest scores it (the two
  `EXCLUDED_FORECAST_DATES` applied, `served_identity` pooled). A date counts if it has at
  least `PT_MIN_ROWS_PER_DATE` (30) rows, as `pesaran_timmermann` already requires.
- **Primary horizons: h=3 and h=7.** Each is judged separately against its own bar. h=14 and
  h=30 are reported, not judged: all-date PT is null at both, and their date counts are far
  below 20.
- **Fixed size, no optional stopping.** Each horizon's window is its **first 20 usable
  forecast dates ≥ 2026-10-05**. It is read once, after the 20th resolves. Later dates are
  ignored. Expected reads: h=3 about 2026-10-29, h=7 about 2026-11-02. Later if forecast
  dates are skipped.
- **Truncation, not extension:** a change to the direction rule, `FLAT_TOLERANCE` or the
  PT code inside the window truncates it at the change. A window under 20 dates is void.

## Bar (fixed now), per primary horizon

All four must hold.

1. `pt_n_dates ≥ 20` in the window.
2. PT excess > 0 with **t ≥ 3.0** (`PT_T_HURDLE`), from `backtest.directional_test.pesaran_timmermann`
   unchanged at the freeze commit.
3. **DA − `realised_down_rate` ≥ 0** (pp, window rows pooled). The 95% CI from a paired
   date-level bootstrap (10,000 draws, seed 0) is reported, not gated.
4. **Drop-one-date guard:** removing the single date with the largest excess leaves PT
   positive with t ≥ 2.0.

**Void, not null:** fewer than 20 usable dates; a truncating change (above); the scorer's
derivation changes mid-window.
**Unresolved, not null:** (2) fails with 2.0 ≤ t < 3.0. Record the achieved MDE. A new
window needs a new prereg declared before its first date resolves.

## Power, stated up front

Back-of-envelope from the exploration slice (h=3: +1.41pp, t=2.67 over 13 dates implies a
per-date SD of about 1.9pp, treating HAC as i.i.d.). At 20 dates the SE is about 0.42pp, so
t ≥ 3.0 needs a mean of about +1.27pp.

- If the true effect is the observed +1.4pp, power is about 60%.
- If it is half that (+0.7pp), power is about 10%.

The +1.4pp comes from a selected slice, so the second case is the likelier one. **A miss at
(2) is probably "unresolved", and should be read as that.** Condition (3) is expected to
fail on current calibration (the model under-calls "down"). That is by design: it stops a
call that loses to always-down from shipping.

## What each outcome buys

| Outcome | Action |
|---|---|
| (1)–(4) pass at a horizon | A reviewed PR converts `served_direction` to a horizon tuple holding **only the passing horizons** (`2026-09-10-served-direction-withheld.md`, "Reopening"). Disclosure copy quotes DA beside `realised_down_rate`. Add an `experiment_log.csv` row. |
| (2) and (4) pass, (3) fails | **Information without a usable call.** Direction stays withheld. Record it, and route the evidence to the ranking-head gate (`next-steps` item 6). Do not recalibrate the call's base rate here; `plans/2026-08-03-direction-prior-correction.md` and `DIRECTION_UPWEIGHT` (neutral, 08-19) already cover that. |
| (2) unresolved | Record the achieved MDE. Direction stays withheld. |
| (2) null or negative | Direction stays withheld. Add a `refuted` or `measured` row to `experiment_log.csv`. Do not re-run without a new prereg and 20 fresh dates. |
| Void | Record it. |

A pass here is a served-panel read, so it is out-of-sample, but it covers a few weeks of one
market regime. The first weeks after any reopening are the real confirmation.

## Instrument

`backend/scripts/measure_direction_reopen.py`, **to be written and unit-tested on synthetic
panels before the first read**, with the same guard as `measure_conformal_pid.py`: it refuses
to read prod for a horizon until that horizon has 20 usable window dates. It takes its records
from `_records_from_frozen_outcomes`, so it cannot drift from the published scorer.

## Amendments

Any amendment is dated and written before the first in-window read.

### 2026-10-30: the cohort is read from the ops partitions, not Postgres

Written before any in-window number is computed. Neither horizon has been read.

- **What changes:** where `forecast_outcomes` is read from. The instrument now hands
  `_records_from_frozen_outcomes` the backtest's own `_panel_read()` (the monthly ops
  partitions in cs2-oracle-data, `db/ops_store.py`) instead of a Postgres session.
- **Why:** prod Postgres hit DiskFull on 2026-10-29 and went read-only on 2026-10-30. Outcomes
  resolved since then are written only to the partitions (PR #122), so a Postgres read cannot
  reach 20 window dates, and the panel JOIN no longer runs there. Separately, since #122 the
  scorer takes a `sql -> rows` reader, and the Postgres loader raised before reading anything.
- **What does not change:** the scorer's derivation (`_records_from_frozen_outcomes`, same
  SQL, same ≥$1 floor, same `EXCLUDED_FORECAST_DATES`), the direction rule, `FLAT_TOLERANCE`,
  the PT code and every number in the bar. The *Truncation* rule names none of these as
  changed, so the window is neither truncated nor void.
- **Evidence the two stores hold the same rows:** the partitions were reconciled to Postgres in
  Phase 1 (Postgres wins every shared row); the nightly aggregate parity check was clean for 15
  nights through 2026-10-28; and the first partition-read backtest (2026-10-30) compared matured
  forecasts per forecast date from both stores and found 0 differences. That last check compares
  row counts, not values. `tests/test_backtest_reads_partitions.py` shows the derivation scores
  identical rows identically from either store. Rows resolved after 2026-10-30 exist only in the
  partitions and are scored by the same derivation.
