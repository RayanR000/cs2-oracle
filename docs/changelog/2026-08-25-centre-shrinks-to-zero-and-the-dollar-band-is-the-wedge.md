# The centre shrinks to zero, and the dollar band's under-coverage is the anchor wedge

**Date:** 2026-08-25
**Scripts:** `backend/scripts/centre_vs_lastprice.py --shrink`,
`backend/scripts/dollar_band_wedge.py` — both read-only, prod Postgres, no replay, no retrain.
**Panel:** 45,285 resolved `>=$1` rows over 20 forecast dates, `excluded_forecast_date` applied.

Two measurements, one on each half of what production serves: where the interval
sits, and what the interval means in dollars.

## A note on the panel source

Both scripts read **prod Postgres by default**, not the Parquet mirror. Per the
`archive-reads` rule neither copy of `ops/forecast_outcomes.parquet` is the full
scored panel, and this was measured rather than assumed: the durable CI-written
archive holds 1–2 fewer clean dates per horizon than prod (h=3: 14 vs 16, h=7:
15 vs 17, h=14: 10 vs 11). Every figure below is the prod read. `--archive-dir`
remains for offline cross-checks.

## 1. The centre's optimal shrinkage is zero

`centre = lambda * r_hat` nests both arms already measured — `lambda=0` is
quoting the last price, `lambda=1` is what production serves — so the
keep/retire question becomes a dial, and a negative skill at `lambda=1` no
longer has to mean the centre is information-free. It does anyway.

| h | dates | lam* | 90% CI | MAE(lam*) | MAE(1) | gain vs served |
|---|---|---|---|---|---|---|
| 3  | 16 | **0.00** | [0.00, 0.00] | 0.03537 | 0.03742 | +5.5% |
| 7  | 17 | **0.00** | [0.00, 0.00] | 0.05025 | 0.05358 | +6.2% |
| 14 | 11 | **0.00** | [0.00, 0.10] | 0.08122 | 0.08853 | +8.3% |
| 30 | 2  | 1.00 | [0.55, 1.00] | 0.13686 | 0.13686 | +0.0% |

The date-bootstrap CI on `lam*` is **degenerate at zero** for h=3 and h=7 and
reaches only 0.10 at h=14: there is no shrinkage of the GBM centre that beats
predicting no move, and serving it unshrunk costs 5–8% of centre accuracy. The
centre is not over-expressed, it is empty.

h=30 is the one cell pointing the other way and has two clean dates inside a
single drawdown. It is not evidence.

## 2. The dollar band under-covers because of the anchor wedge, and that is fixable

`predict()` quotes the triple off `current_price` — the smoothed anchor,
substituted **unconditionally** at `forecaster.py:6443` and so a number no venue
quoted — while the outcome resolves off `base_price`. `_derive_verdict` already
reports both predicates; nothing had attributed the gap between them.

| h | rows | dollar cov | calibrated cov | gap | 90% CI | recoverable | genuine |
|---|---|---|---|---|---|---|---|
| 3  | 15,700 | 0.721 | 0.887 | +0.167 | [+0.029, +0.317] | 70.0% | 30.0% |
| 7  | 16,828 | 0.722 | 0.904 | +0.182 | [+0.088, +0.278] | 72.6% | 27.4% |
| 14 | 10,774 | 0.540 | 0.828 | +0.288 | [+0.182, +0.394] | 66.8% | 33.2% |
| 30 | 1,983  | 0.657 | 0.771 | +0.113 | [+0.093, +0.135] | 35.6% | 64.4% |

**A consumer reading the h=14 band in dollars gets 54% coverage from an interval
calibrated to 80%.** Two thirds of those misses are inside the rebased band —
the wedge alone, recoverable by re-anchoring the served quote.

The decile cut is the attribution, with the calibrated column as its own
control:

| decile | \|wedge\| <= | rows | dollar cov | calibrated cov |
|---|---|---|---|---|
| 1 | 0.0046 | 18,114 | 0.906 | 0.906 |
| 2 | 0.0218 | 4,530 | 0.877 | 0.888 |
| 3 | 0.0506 | 4,527 | 0.802 | 0.874 |
| 4 | 0.0928 | 4,531 | 0.610 | 0.838 |
| 5 | 0.1387 | 4,526 | 0.466 | 0.842 |
| 6 | 0.1918 | 4,530 | 0.280 | 0.835 |
| 7 | 0.5542 | 4,527 | 0.093 | 0.839 |

Dollar coverage falls **0.906 → 0.093** across the wedge while the calibrated
band stays flat at 0.84–0.91. A width or calibration defect would have moved
both columns; only the published one moves. On the 40% of rows where the two
anchors agree (decile 1) the predicates are identical, as they must be.

**Limit.** The stored quote is the SMOOTHED one, so this measures
quote-vs-resolved-anchor and bundles the unconditional smoothing substitution
(what Arm A gates) with ordinary drift between the two resolution times. It is a
CEILING on what any anchor arm can buy, not a simulation of one — that needs a
replay against the raw price frame.

## 3. Arm A cannot collect most of that ceiling — corrected same day

Arm A (`SERVE_OUTLIER_GATED_ANCHOR`, already built, off by default) gates the
smoothing substitution on the >10% outlier test, so it serves the RAW quote for
non-deviating items and keeps smoothing the deviating ones. Splitting the
recoverable misses by wedge size asks what that leaves it:

| h | dollar misses | recoverable | of those, \|wedge\| <= 10% | > 10% |
|---|---|---|---|---|
| 3  | 4,387 | 3,073 | 16.0% | **84.0%** |
| 7  | 4,682 | 3,399 | 16.9% | **83.1%** |
| 14 | 4,954 | 3,311 | 23.0% | **77.0%** |
| 30 | 680   | 242   | 45.9% | 54.1% |

**77–84% of the recoverable gap sits at wedges above 10%** — the items where Arm
A goes on smoothing. It can address roughly a sixth to a quarter of the ceiling,
not the whole of it. (The mapping is directional, not exact: the wedge is
base-vs-quote across time while the arm's test is latest-vs-3d-median at the
anchor.)

## 4. And the replay harness cannot measure the arm's effect on this at all

Control and arm at `REPLAY_ANCHOR=2026-07-15` (same artifact, 971 vs 970 items,
879 deviating):

| h | control cov | Arm A cov | control halfw | Arm A halfw |
|---|---|---|---|---|
| 3  | 90.63% | 94.85% | 7.97% | 7.92% |
| 7  | 90.94% | 92.58% | 11.44% | 11.32% |
| 14 | 90.32% | 92.06% | 16.73% | 16.68% |
| 30 | 87.54% | 88.76% | 24.79% | 24.54% |

The arm moves coverage +1.2 to +4.2pp at unchanged width — but note the level:
**both arms sit at 87–95% against a nominal 80%**, nowhere near the 54–72% the
published dollar band actually delivers. That is not a discrepancy to reconcile,
it is the harness: `_resolve` resolves the outcome on the SAME smoothed basis
`predict` quotes from, so the replay has no wedge by construction and cannot see
the defect. Raising an already over-covering band is not obviously a gain
either.

Measuring an anchor arm's true effect needs the production pairing — band on the
served quote, outcome on `resolve_anchors` — which today exists only by waiting
for rows to mature.

## 5. Which side of the mismatch is at fault — scoped

`_smoothed_anchor_prices` (serving) and `resolve_anchors` (scoring) are the SAME
definition: median of the last 3 observations at or before the anchor, span
bounded at 7 days. Both anchor on the same date, since `forecast_date` IS the
serving anchor date. So the wedge should not exist at all, and there are only
two ways it can: the archive is REVISED after serving, so the scorer resolves
against data the serving path never saw, or the serving path is not applying
the definition it claims.

Re-resolving the panel's anchors today with `resolve_anchors` itself, against
the real voted frame from `fetch_price_history` (546,907 rows / 5,536 items;
11,304 of 17,849 panel anchors resolvable within span):

| comparison | median \|diff\| | p90 | share > 1% |
|---|---|---|---|
| resolve_anchors (today) vs stored `base_price` | **0.0220** | 0.2418 | 56.6% |
| resolve_anchors (today) vs stored `current_price` (served quote) | **0.0759** | 0.2333 | 89.0% |
| stored `base_price` vs served quote | 0.0587 | 0.2133 | 69.8% |

**The serving side is the larger error.** Today's resolver reproduces the stored
`base_price` to a median 2.2% while sitting 7.6% away from the quote the same
row was served on — so the scoring leg is roughly where it says it is, and the
served quote is the leg that drifts. Archive revision is real but the smaller
term (the 2.2% residual is what a month of revisions moved).

That points the fix at the serving side, which is where arms B and C already
live, and gives them a target: bring the served quote within ~2% of what
`resolve_anchors` would say, which is the floor revision alone imposes.

A caution on the 63% match rate: anchors the resolver drops today are exactly
the thin/stale ones, so this table describes the resolvable majority and may
understate the tail.

**A method note worth keeping.** The first attempt at this hand-rolled the
consensus as `median(mean_price)` over the universe filter and produced 17-22%
disagreement with BOTH stored legs — an artefact, not a finding. `AGENTS.md`
invariant 2 and `_outcomes`' docstring both say why: a plain median lets BUFF's
bid and Steam's trailing-window means vote, and neither is in the consensus
`predict` quoted from. Use `fetch_price_history` and `resolve_anchors`, never a
hand-rolled aggregate.

## 6. Arm B's premise does not survive — the quote is not a window problem

Arm B shrinks `SMOOTH_WINDOW` (3→2) or the span (7→3), which assumes the served
quote IS the median of a window and the window is the wrong size. Before
touching a constant shared with the backtest resolver, that premise was tested:
six candidate anchors were rebuilt from the voted frame at each row's
`forecast_date` and compared against the stored served quote (17,801 rows).

| candidate | vs SERVED quote (median) | matches quote <1% | vs base (median) |
|---|---|---|---|
| last raw obs        | 0.0640 | 11.5% | 0.0417 |
| median last 3, 7d   | 0.0667 |  8.7% | **0.0262** |
| median last 3, no span | 0.0616 | 11.7% | 0.0366 |
| median last 2, 7d   | 0.0688 |  7.8% | 0.0316 |
| exact-day price     | 0.0935 |  3.0% | 0.0456 |
| median last 5, 14d  | 0.0588 | 11.3% | 0.0402 |

**Nothing reproduces the served quote** — the best candidate matches it on 11.7%
of rows — while the shipped definition reproduces `base_price` to 2.6%. No
setting of the window recovers a number that is not a function of this frame at
all. Arm B would be tuning a parameter that is not the cause.

## 7. The likely cause: the serving path anchors on a day still being written

The archive says when each row arrived:

| ingest lag | rows (since 2026-07-20) |
|---|---|
| 0 days | 4,256,407 |
| 1 day  | 1,246,091 |

and for a single day it is starker — **every** row stamped `day = 2026-08-04`
(300,321 rows, 10 sources, 37,910 items) was ingested LATER, on 08-05.

That lands exactly inside the serve window. The chain runs Aggregator (23:00
UTC) → Price Forecast, so the forecast quotes from a frame whose newest day is
still filling in as it runs; the scorer resolves the same `forecast_date` days
later, against the completed day. Same definition, same date, different data —
which is precisely the signature above: the served quote matches no
reconstruction, while the resolver reproduces the scoring leg.

**Stated as the leading hypothesis, not a proof.** The archive holds no
serve-time snapshot, so the frame `predict` actually saw cannot be reconstructed
from it. The decisive test is cheap instrumentation rather than more analysis:
log the anchor's inputs (observation dates, source count, per-source prices) on
the serving path, and one night's forecast settles it.

## 8. Instrumented — and the first read already narrows it

`ItemForecaster._audit_serving_anchor` now runs on the serving path, before
`_serving_base_price` overwrites `price` with the served base (after it, the
audit would record the answer instead of the inputs — pinned by a test that
reads the call order out of `predict`'s source). It logs, unconditionally, how
much of the frame reaches the anchor day and the distribution of each item's lag
to its newest observation; `ANCHOR_AUDIT=1` adds a per-item Parquet carrying the
raw quote and the smoothed value SEPARATELY, which is exactly the pair that
cannot be reconstructed afterwards. Every failure is swallowed: a forecast must
not die because a diagnostic could not write.

First read, a local predict against the canonical archive:

```
Anchor audit @ 2026-08-24: 5,536 of 16,608 frame rows land ON the anchor day;
item lag to newest obs median 0d, p90 0d, 100.0% current
Anchor audit — rows per day (last 5): 2026-08-22=5,536, 2026-08-23=5,536, 2026-08-24=5,536
```

**Every item is current.** All 5,536 have an observation on the anchor day, and
the lag distribution is 0 at the p90. So §7's mechanism is not "items are
missing from the frame" — at least not for a run executed hours after ingest
completes, which is what this was.

What it does NOT settle is the CI timing, which is the version of the hypothesis
that matters: the chain runs the forecast immediately after the 23:00 aggregator,
and every row stamped `day = 2026-08-04` arrived on 08-05. This read is from a
run at 13:48, long after that day closed. The audit is now in place to catch the
23:00 run, which is the one that has to be seen.

The frame's depth is worth noting for whoever reads that log next: 16,608 rows is
exactly 5,536 x 3, the tail retained by chunked engineering. The anchor's
median-of-3 window and the retained tail are the same three rows, so a gap in an
item's history reaches the anchor through the span bound rather than through row
count.

## 9. The serving math is EXACT — so the wedge is a data change, not a code defect

The audit's per-item dump makes the decisive test possible for the first time:
compare the serving path's own smoothed value against `resolve_anchors` on the
SAME archive, item by item. 5,536 items at anchor 2026-08-24, all resolvable:

| comparison | median \|diff\| | p90 | match <0.1% |
|---|---|---|---|
| serving `_smoothed_price` vs `resolve_anchors` | **0.00000** | **0.00000** | **100.0%** |
| serving raw quote vs `resolve_anchors` | 0.00000 | 0.14286 | 54.6% |

**The two legs agree exactly, on every item.** Given the same data, the serving
path computes precisely what the scorer computes — the smoothing is not
mis-implemented, the window is not mis-sized, and the substitution is not
introducing the wedge.

That leaves one explanation for the 7.6% historical gap: **the data itself
changed between serve time and scoring time.** The stored `current_price` was
computed against a version of those days that the archive no longer holds, which
is also why no reconstruction from today's archive could match it (§6) while
`base_price` — resolved later, closer to the current version — sits within 2.2%.

It also retires all three anchor arms as candidates. A, B and C each change how
the serving path *computes* the anchor; §9 says that computation is already
identical to the scorer's. None of them addresses a moving input.

**And the revision cannot be audited retrospectively.** `cs2-oracle-data` is
published as an orphan commit and force-pushed on every run, so `main` is a
one-commit history and no previous version of any day survives. There is no
diff to take. That property was adopted to keep Parquet blobs out of git
history, and it is exactly what makes this class of question unanswerable after
the fact.

## What follows

1. **Do not build any of the three anchor arms.** A reaches a sixth to a
   quarter of the ceiling (§3), B tunes a window that does not describe the
   served quote (§6), and §9 shows the serving computation already matches the
   scorer exactly. All three fix the arithmetic; the arithmetic is right.
2. **Persist the serve-time quote instead.** The wedge is a moving input, so the
   only way to make the published band and the score agree is to freeze what was
   served: `ANCHOR_AUDIT=1` already writes the raw quote and the smoothed value
   per item per anchor. Turning it on in `price-forecast.yml` and publishing the
   Parquet alongside `item_forecasts` makes every future wedge attributable — and
   makes scoring against the served basis possible rather than approximate.
3. **The archive keeps no history, and that is now a measured cost.** The orphan
   force-push means no previous version of any day exists, so revisions cannot
   be diffed and this question could only be answered going forward. Worth
   deciding deliberately whether the daily publish should retain even a shallow
   history.
4. **The centre should be shrunk to zero at h<=14**, which is the same change as
   retiring the GBM's centre, arrived at independently.
5. Neither is quotable yet under `MIN_FORECAST_DATES = 20` — h=3 is at 16 dates,
   h=7 at 17, h=14 at 11. Signs and the decile monotonicity are what these
   establish; re-read the magnitudes at 20.
