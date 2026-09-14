# 2026-09-13 — Archive-basis served-rank leg: pre-registered, instrumented, VOID on power until ~09-27

**Status:** ARMED, not run. The power gate is measured and fails (13 qualifying dates vs 20).
The instrument refuses to compute a statistic below the gate and exits 1.
**Prereg:** `docs/research/2026-09-13-archive-basis-centre-rank-preregistration.md`.
**Follows:** `2026-09-13-clean-era-centre-measured.md` (the +0.094 observation) and
`2026-09-13-served-centre-rank-void.md` (the quote-basis VOID).

## What this closes out

The clean-era read left exactly one named follow-up — *does h=14 tied ranking survive on
served 2026 dates?* — and required a new prereg. The first attempt VOIDed on a basis defect,
not a calendar one. This entry writes the prereg that fixes the basis, builds the instrument,
and measures the gate. No rank statistic has been computed on the archive basis.

## Three panel facts measured while scoping (counts, not results)

1. **A third ~100%-frozen day exists, and it is inside the served era.** On the served cohort
   (1,114 slugs, consecutive days only) the exactly-unchanged share runs **0.08–0.19** through
   2026-07-11..09-08, with exactly three blow-outs: **07-16 (1.000)**, **07-22 (1.000)** and
   **08-23 (0.995)**. 07-16/07-22 were already catalogued; **08-23 was not**. The next highest
   is 08-19 at 0.578, so the inherited 0.70 threshold separates cleanly without being tuned here.

2. **The archive-vs-quote anchor gap is era-split, and pre-08-11 it is larger than the signal.**
   Median |gap| over 21,688 panel rows is 0.051, but per-date: **0.089–0.240** for
   2026-07-17..08-09 and **0.013–0.029** for 2026-08-11..08-27. At a ~7% h=14 return scale the
   earlier era cannot support an archive-anchored read at all. This is why the primary cohort is
   `anchor_clean == True` and not the previous leg's 70%-coverage rule — a design choice forced
   by a measurement, fixed ahead of data.

3. **Archive holes cost 8 of 24 dates their exact `d+14` anchor** (missing 08-28, 08-30, 08-31,
   09-01, 09-03, 09-04, 09-05). Hence the pre-registered ±2-day nearest-non-frozen anchor with
   ties toward the earlier day, plus an exact-only sensitivity read.

## Design decisions fixed ahead of data

- **All three legs anchored on the contemporaneous archive voted composite** — `r_hat` included,
  so the model is charged for the anchor wedge. This is the direct answer to the VOID note's
  lesson that quote-basis outcomes cannot referee ranking (`-r1` IC flips +0.043 → −0.138 purely
  by swapping the outcome leg; the quote/voted gap is +0.60 rank-correlated with trailing 1d
  returns, and the 2025 edge is built on reversal features).
- **Frozen exclusion is whole-DAY, never row-level.** Dropping frozen rows selects volatile items
  and flipped the h=3 climatology verdict spuriously; barring a dead anchor day is cohort-neutral.
- **VOID is absolute.** Below 20 qualifying dates `evaluate_bars` returns no statistic at all, so
  a VOID cannot be quietly re-read as "underpowered but suggestive" — which is exactly how the
  previous leg's 13-date exploratory cell (±0.10 CIs) invited misreading.

## Gate

`--gate`, prod Postgres read-only, 2026-09-13: h=14 panel is 24 dates / 22,705 rows after
`excluded_forecast_date`; `anchor_clean` is non-null on 48.2% of rows and qualifies
**13 dates** (2026-08-11 .. 08-27). **13 < 20 → VOID, do not run.**

**When it clears.** The core chain is active again and `item_forecasts` carries 22,144 rows/day
for 2026-09-06..09-13 after an 08-28..09-05 outage. Those resolve at h=14 on 2026-09-20..09-27,
reaching 20 qualifying dates **on or about 2026-09-27** if the chain stays green.

## Instrument

`backend/scripts/archive_basis_centre_rank.py` (+ `tests/test_archive_basis_centre_rank.py`,
19 tests). Read-only; imports the panel from `served_centre_rank` and the interval helpers from
`clean_era_centre_ab` — no second implementation of either. `frozen_anchor_dates` reproduces the
ad-hoc measurement exactly on real data (07-16/07-22/08-23 at 0.70; adds 08-19 at 0.50).

## Reproduce

```
venv/bin/python -m scripts.archive_basis_centre_rank --gate
venv/bin/python -m pytest tests/test_archive_basis_centre_rank.py -q
```

No serving change. The range stance and the withheld direction are untouched.
