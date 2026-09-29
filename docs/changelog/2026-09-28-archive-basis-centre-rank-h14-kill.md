# Archive-basis h=14 served-rank leg: KILL

**Date:** 2026-09-28
**Prereg:** `docs/research/2026-09-13-archive-basis-centre-rank-preregistration.md`
**Instrument:** `backend/scripts/archive/archive_basis_centre_rank.py` (19 tests pass)
**Status:** KILL per the prereg. No serving change.

## Gate

`--gate` against prod Postgres (read-only) printed **RUNNABLE: 20 qualifying `anchor_clean`
dates (need 20)**: 2026-08-11 → 09-13, with 08-12 and 08-21..23, 08-28..09-06 absent. That is
the whole margin, so the leg was read on the day it cleared.

**Archive:** the durable origin, `RayanR000/cs2-oracle-data@21180a1` (2026-09-28T20:29Z),
`prices-2026-07..09.parquet`, fetched into a scratch directory. Neither local copy was usable:
the working copy ends 09-08, the local data checkout ends 09-18, and the latest forward anchor
needs 09-27. Durable is missing 07-27, 07-30, 08-02/03, 08-28, 08-30..09-01, 09-03..05 and 09-19.

## Result (primary cohort `anchor_clean`, 20 dates, 7,310 rows)

| statistic | mean | 95% CI (date-block bootstrap) |
|---|---|---|
| model IC | +0.0890 | [+0.0611, +0.1174] |
| naive IC (`−return_1d`) | +0.0634 | [+0.0331, +0.0940] |
| **paired model − naive** | **+0.0255** | **[−0.0051, +0.0546]** |

Bar: CONFIRMED needs both CIs entirely positive. The model's IC clears, but the paired CI
spans zero, so the verdict is **KILL**. On a wedge-free basis the q50 centre ranks items at h=14,
but it cannot be told apart from a one-line reversal baseline. This is the same finding as
`model-loses-to-minus-return-1d`, now on the archive basis at 20 dates.

## Sensitivities (reported, never barred on)

| read | primary-cohort dates | primary verdict | wedge cohort paired | all cohort paired |
|---|---|---|---|---|
| exact `d+14` only | 15 | VOID (<20) | +0.034 [−0.013, +0.090] | +0.026 [−0.020, +0.084] |
| frozen threshold 0.50 | 19 | VOID (<20) | +0.028 [−0.014, +0.072] | +0.015 [−0.029, +0.063] |

Both sensitivities VOID on the primary cohort, so no primary statistic is read from them. In
every secondary cohort under all three reads the paired CI spans zero, so nothing points the
other way. The 0.50 read bars 08-19, as the prereg predicted.

Full output: `docs/research/data/2026-09-28-archive-basis-centre-rank-h14-{primary,exact,frozen050}.json`.

## Consequences

- Item 6 (ranking-head transfer) loses its archive-basis support. `RANKING_HEAD` stays a sidecar,
  and no rank goes into band or direction.
- The prereg's reproduce block cites `scripts.archive_basis_centre_rank`. The module is now
  `scripts.archive.archive_basis_centre_rank`.

## Reproduce

```
cd backend
venv/bin/python -m scripts.archive.archive_basis_centre_rank --gate
venv/bin/python -m scripts.archive.archive_basis_centre_rank --archive-dir <durable prices dir> --out /tmp/abcr_h14.json
venv/bin/python -m scripts.archive.archive_basis_centre_rank --archive-dir <dir> --exact-only
venv/bin/python -m scripts.archive.archive_basis_centre_rank --archive-dir <dir> --frozen-threshold 0.50
```
