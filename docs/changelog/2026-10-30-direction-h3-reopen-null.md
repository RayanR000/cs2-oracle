# Served direction at h=3: the preregistered reopening read is null

**Date:** 2026-10-30
**Prereg:** `research/2026-10-08-direction-reopening-preregistration.md` (with its 2026-10-30
amendment: the cohort is read from the ops partitions)
**Script:** `backend/scripts/measure_direction_reopen.py` at main `00ff1ae`, data repo `77fbc28`
**Status:** final for h=3. The window is fixed, so a re-run returns the same answer. Direction
stays withheld. h=7 is not yet read.

## Read

First 20 usable forecast dates ≥ 2026-10-05: **2026-10-05 .. 2026-10-24**, 19,927 rows (≥$1,
`EXCLUDED_FORECAST_DATES` applied, served identities pooled).

| Condition | Bar | h=3 | Holds |
|---|---|---|---|
| (1) dates | ≥ 20 | 20 | yes |
| (2) PT excess, t | > 0, t ≥ 3.0 | +0.05pp, t = 0.14 | no |
| (3) DA − realised down | ≥ 0 | 32.9% − 49.8% = **−16.9pp** (95% CI −21.4, −12.6) | no |
| (4) drop-one-date | t ≥ 2.0 | dropped 10-11: −0.10pp, t = −0.36 | no |

Achieved MDE: 1.05pp.

**Verdict: NULL.** t is below the 2.0 floor for "unresolved", so this is not an underpowered
miss. The +1.41pp, t = 2.67 exploration slice (≥ 09-22, through 10-04) did not carry into fresh
dates, as the prereg's power note expected. The call also loses to always-down by 16.9pp,
wider than the 11.9pp seen in that slice: the model still under-calls "down".

## Action (prereg outcome table, "null or negative")

- `served_direction` stays withheld at h=3.
- `experiment_log.csv` row added (`refuted`).
- No re-run without a new prereg and 20 fresh dates.

h=7 had 18 of 20 window dates at this read (the script refused it and computed nothing).
Expected read ~2026-11-02.

## Reproduce

From `backend/`, with the data repo's `price-archive` linked at the repo root:

    venv/bin/python -m scripts.measure_direction_reopen
