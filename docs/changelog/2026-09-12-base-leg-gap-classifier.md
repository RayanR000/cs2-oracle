# Gap classifier counts the base leg too (2026-09-12)

Backtest Accuracy failed 2026-09-11/12 (runs 34549459944, 34691910482) at
"resolution rate: 5,536 of 22,144 / 27,680 newly-resolvable forecasts
(25.0% / 20.0%)". All 5,536 are the h=3 cohort dated 09-06 — the first
forecast date after the 08-28..09-05 collection holes — reproduced locally
read-only against prod (full h=3 group load: 105,374 anchors / 99,817
resolved, matching CI exactly). The cell drops 5,536 of 5,536 on `base_none`
while every one of its target anchors resolves: the base window [08-30, 09-06]
holds exactly {09-02, 09-06} where `resolve_anchors` requires 3 observations
within 7 days (08-29 sits 8 back).

`classify_archive_gap` only counts the actual leg's forward window, which here
reads clean (09-07/08/09 all present), so a permanent, deterministic,
per-item-universal failure counted FRESH. Permanent because history is fixed:
the cell re-enters `to_resolve` forever and taxes every future fresh rate
until dilution. Not a 09-12 resolver regression — no resolver change shipped
in that window, and the resolver is correct to drop these anchors. Not one
horizon bucket either: 27,680 is 5 fresh (date, horizon) cells, of which this
is one.

`backtest.resolution_gate.classify_base_gap` counts the base range
`[f - staleness, f]` (both ends inclusive, matching the resolver's strict `>`
comparison) on the same positive-evidence standard: only a globally missing
range is excused, item-level sparsity still counts FRESH. `backtest_accuracy.py`
ORs it with the actual-leg check. Behaviour is unchanged everywhere else —
09-07/08 h=3 hold 3/4 base days and still resolve, as does every other fresh
cell, verified against the prod archive.

Projected gate on the failing cohort: fresh rate 0.0% over 22,144 informative
attempts, coverage 0.04%, gap 66,474 (12.6%) warned on — green, with
22,144 outcomes accruing (08-11 h=30, 08-27 h=14, 09-07/08 h=3). That
unfreezes forecast_outcomes (stuck at 437,380 since the 09-10 success) and
with it the q_hat calibration panel, the ANCHOR_AUDIT read, the Arm A
coverage A/B, and exceedance maturation.
