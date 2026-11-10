# Served direction at h=7: information without a usable call

**Date:** 2026-11-09
**Prereg:** `research/2026-10-08-direction-reopening-preregistration.md` (with its 2026-10-30
amendment: the cohort is read from the ops partitions)
**Script:** `backend/scripts/measure_direction_reopen.py` at main `cfadfda`, data repo `f9077c2`
**Status:** final for h=7. The window is fixed, so a re-run returns the same answer. Direction
stays withheld. h=3 was read 2026-10-30 (null).

## Read

First 20 usable forecast dates ≥ 2026-10-05: **2026-10-05 .. 2026-10-24**.

| Condition | Bar | h=7 | Holds |
|---|---|---|---|
| (1) dates | ≥ 20 | 20 | yes |
| (2) PT excess, t | > 0, t ≥ 3.0 | +1.78pp, t = 3.71 | yes |
| (3) DA − realised down | ≥ 0 | 44.8% − 57.6% = **−12.8pp** (95% CI −16.3, −9.5) | no |
| (4) drop-one-date | t ≥ 2.0 | dropped 10-13: +1.63pp, t = 3.75 | yes |

Achieved MDE: 1.44pp.

**Verdict: INFORMATION_WITHOUT_USABLE_CALL.** The up/down calls carry information against their
own marginals, and it does not hinge on one date. The call itself still loses to always-down by
12.8pp, so acting on it would lose.

Report-only, never judged: h=14 PT +2.95pp (t = 10.0), DA 43.6% vs always-down 60.0%
(−16.4pp). h=30 had 5 of 20 window dates.

## Action (prereg outcome table, "(2) and (4) pass, (3) fails")

- `served_direction` stays withheld at h=7.
- `experiment_log.csv` row added (`measured`).
- Evidence routed to the ranking-head gate (`next-steps` item 6).
- The call's base rate is not recalibrated here (`plans/2026-08-03-direction-prior-correction.md`
  and `DIRECTION_UPWEIGHT` already cover that).
- No re-run without a new prereg and 20 fresh dates.

## Reproduce

Log: `direction-reads/2026-11-09-h7-read.log` (outside the repo). From `backend/`, with the data
repo's `price-archive` linked at the repo root:

    venv/bin/python -m scripts.measure_direction_reopen
