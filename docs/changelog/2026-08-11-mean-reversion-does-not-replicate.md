# The index mean-reversion sign was window-specific; N2's own-history leg is closed

**Pre-registered:** `docs/research/2026-08-11-mean-reversion-preregistration.md`, committed and
pushed as `c3c1284` before the test window was scored.
**Window:** 2014-01-01 → 2023-12-31, disjoint from the 2024-2026 window that produced the
hypothesis. Index: 3,652 days, **3,648 valid**, median 215 items/day, daily log-return sd 0.02373.

## The result: both rules fail

| h | n (non-ovl) | corr | contrarian DA% | constant call% | edge |
|---|---|---|---|---|---|
| 3 | 1,155 | −0.038 | 44.7 | 56.9 | **−12.2** |
| 7 | 494 | +0.008 | 40.9 | 60.9 | **−20.0** |
| 14 | 246 | +0.038 | 40.7 | 61.8 | **−21.1** |
| 30 | 114 | +0.016 | 42.1 | 64.0 | **−21.9** |

**PRIMARY** (corr < 0 at ≥ 3 of 4): **1 of 4 — FAIL.**
**SECONDARY** (contrarian beats the constant call at ≥ 3 of 4): **0 of 4 — FAIL.**
No horizon was void; the smallest cell carries 114 non-overlapping windows.

The correlations are indistinguishable from zero on roughly **2,000 non-overlapping windows**,
against the 437 that produced the −0.093 to −0.162 readings. The discovery window's 4-of-4
negative correlation does not survive contact with an independent decade.

## Committed interpretation, applied

The pre-registration's third branch fires verbatim: *"the 4-of-4 negative correlation was
window-specific. N2's own-history leg closes completely, and the exogenous leg is all that
remains of the track."*

**N2's own-history leg is closed.** Neither the drift nor its negation carries information about
the forward market factor. Combined with
`2026-08-11-market-factor-is-not-forecastable-from-its-own-history.md`, the market term has no
serveable forecast from index history at any horizon, in either direction, across twelve years.

## Why this entry exists at all

The pattern that motivated it was found post-hoc on a read designed to test the opposite sign,
and it was recorded rather than acted on for that reason. Without the pre-registration, negating
a refuted estimator and reporting the discovery window's −0.09 to −0.16 would have read as the
project's first date-level signal. It is not one. This is the second time today a pre-registered
bar converted an attractive number into a decision — the first was C1
(`2026-08-11-c1-fails-the-clean-cohort-read.md`).

## One observation, explicitly not a finding

The date-level constant call is strong and grows with horizon — 56.9 / 60.9 / 61.8 / 64.0% on
2014-2023. The market factor's *direction* is highly predictable by a majority call; its
*magnitude and timing* are not predictable by these estimators. Nothing here says whether a
different estimator family could do better, and the exogenous leg (FX, events, player counts)
remains untested.

## Reproduce

`n2_mean_reversion_test.py`, kept out of the repo for the same reason as the first read: it is a
one-shot analysis over already-tested library code (`tests/test_market_factor.py`), not a harness
anything should depend on.
