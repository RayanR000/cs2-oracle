# Served h=7 band vs a naive empirical-quantile band: the preregistered read, served better

**Date:** 2026-11-09
**Prereg:** `research/2026-10-08-naive-quantile-band-h7-preregistration.md` (A1: panel read from the
ops partitions, not the single-file mirror)
**Script:** `backend/scripts/outside_baseline.py`, data repo `f9077c2`; logs and JSON in
`direction-reads/2026-11-09-h7-naive-band-*` (outside the repo)
**Status:** final. The window is fixed. No re-read is scheduled.

## Read

First 20 scoreable h=7 forecast dates ≥ 2026-09-27: **2026-09-27 .. 2026-10-16**, 16,893 rows
(≥ $1), 1,066 items, statsforecast 2.1.1.

| Arm | cover | median half-width | mean IS (α = 0.2) |
|---|---|---|---|
| S, served | 77.1% | 6.89% | 0.2429 |
| N, naive (candidate) | 88.7% | 8.88% | 0.2459 |
| E, AutoETS (report only) | 75.0% | 6.37% | 0.2533 |

Primary, S − N per-date interval score: **−0.0030, 90% CI [−0.0060, −0.0001]** (negative = served
better). S − E: −0.0105 [−0.0149, −0.0058].

| Bar | Result |
|---|---|
| 1. IS win: primary > 0 and CI lower > 0 | **fails**: primary is negative |
| 2. Coverage guard: \|cov_N − 80\| ≤ \|cov_S − 80\| + 2pp | fails: N 8.7pp off, S 2.9pp off |
| 3. Era-split: S − N > 0 in both sub-windows | **fails**: −0.0079 [−0.0104, −0.0052] (09-27..10-03, 7 dates), −0.0004 [−0.0043, +0.0033] (10-04..10-16, 13 dates) |

Checks: window 20 scoreable dates; constants unchanged against `997e31c` (the diffs are line
wrapping in `scoring.py` and the A1 storage swap in `load_panel`, same filters); row loss 61,800 of
64,565 = 95.7% (≥ 95%).

**Verdict: SERVED BETTER.** The CI's upper bound is below zero, by a hair (−0.0001): the margin is
small and the second sub-window alone is a tie. N fails bars 1, 2 and 3, so the line closes
either way.

Report-only: h=3 S − N +0.0027 [−0.0003, +0.0057] (tie), h=14 −0.0257 [−0.0359, −0.0158], h=30
13 dates, not judged.

## Action (prereg outcome table, "SERVED BETTER")

- Close the naive-band line at h=7. `experiment_log.csv` row added (`refuted`).
- Serving is unchanged. No design spec.
