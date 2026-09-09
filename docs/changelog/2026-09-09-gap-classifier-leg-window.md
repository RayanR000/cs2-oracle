# Gap classifier counts the leg window, not the horizon (2026-09-09)

Backtest Accuracy failed 2026-09-08/09 (runs 34225582469, 34351633071) at
"resolution rate: 16,626 of 116,304 newly-resolvable forecasts (14.3%)".
All 16,626 are h=30 forecasts dated 08-05/06/07 targeting 09-04/05/06 —
reproduced locally read-only against the working archive (max_day 2026-09-08,
same as CI): 16,680 `actual_none` + 17 `base_none`, of which 71 are chronic
(2025-12-01 vintage) and 16,626 fresh, matching CI exactly.

Diffing the resolver against the archive dates: the 2026-08-28..09-05
collection holes (08-28, 08-30, 08-31, 09-01, 09-03, 09-04, 09-05) leave exactly
2 globally-covered days in the 7-day window the actual leg may draw from, where
`resolve_anchors` requires SMOOTH_WINDOW=3 within MAX_WINDOW_SPAN_DAYS=7. No
item could resolve through it — per-item voted series are complete on both
existing days (16,642 of 16,680 hold exactly 2); the days do not exist. The
resolver is correct to drop them.

`classify_archive_gap` counted covered days over the whole 30-day
`(f_date, target_date]` horizon (23-24 present), which coincides with the leg's
effective range only while the horizon fits inside the staleness bound — the
category was built for the h=3 08-02/03 shape, where the two are identical.
Past the bound the count is calendar, not leg. The classifier now takes
`staleness_days` and counts only `f < day <= target` with
`day >= target - staleness` — the days `resolve_anchors` can actually select
(older fails the anchor-staleness rule, at-or-before fails the disjoint guard;
the floor is inclusive, matching the resolver's strict `>` comparison).
`backtest_accuracy.py` passes MAX_WINDOW_SPAN_DAYS. Behaviour below the bound
is byte-identical; above it the category strictly widens, and only on positive
evidence of missing days — the loophole properties (warn every run, gap leaves
the fresh denominator, whole-cohort gap still fails) are untouched.

Projected gate on the failing cohort: coverage 0.14%, fresh rate 0.08% —
green with gap + chronic warnings. The 09-03 boundary cohort (leg holds
exactly {08-27, 08-29, 09-02}) still resolves and is still scored, which pins
the inclusive floor against production data rather than assertion.

Upstream note, not fixed here: 7 of 11 days 08-28..09-05 are permanently
missing (collector writes yesterday's prices; CSGOTrader serves only /latest/,
so there is no backfill). The gap warning now attributes them every run. If
the holes keep coming at this rate the warn line, not the gate, is where it shows.
