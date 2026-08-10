# Steam's trailing-window means have been voting against point-in-time asks

**Date:** 2026-08-09
**Plan:** `.superpowers/sdd/2026-08-09-label-integrity/task-4-brief.md`, Task 4 of the
label-integrity branch (last of four; `BID_SOURCES` exclusion, `n_ask_sources`, and the
composition-stability measurement landed first).
**Change:** `backend/models/item_parser.py` (new `TRAILING_WINDOW_SOURCES`),
`backend/models/forecaster.py` (exclusion in `_apply_multi_source_voting`,
`VOTED_CACHE_VERSION` 5 → 6, both constants re-exposed as `ItemForecaster` class
attributes), `.github/workflows/price-forecast.yml` (cache key `voted-v4-` → `voted-v6-`),
`backend/tests/test_trailing_window_sources.py` (new, 6 cases),
`backend/tests/test_bid_source_voting.py` (2 fixtures updated — see below).
**Suite:** `test_trailing_window_sources.py` + `test_n_ask_sources.py` +
`test_bid_source_voting.py`: 24 pass. Also re-ran `test_merge_17mafo_gap.py`,
`test_predict_tail_truncation.py`, `test_forecaster.py`, `test_phase_collapsed_universe.py`,
`test_phantom_slug_universe.py`, `test_ab_harness_universe.py`: 280 pass, 0 failures.

## What was voting

`aggregator_steam_7d`, `aggregator_steam_30d`, `aggregator_steam_90d` are written
straight from csgotrader's `last_7d`/`last_30d`/`last_90d`
(`collectors/csgotrader_aggregator.py:294-312`) — MA(7)/MA(30)/MA(90) of Steam's own
**sale** price, not a point-in-time ask. `ItemForecaster._apply_multi_source_voting`
medians them against genuine asks on equal terms, with no basis tag, on every training
label and every scored outcome in this project.

## The mechanism — corrected

The task brief this was drafted from framed these feeds as sitting *below* live asks,
by analogy with the bid (`aggregator_buff163_buy`, excluded 2026-08-07). That analogy is
wrong on the level. Measured against genuine third-party asks — `buff163`, `csfloat`,
`csmoney`, `skinport`, `youpin`, with the Steam-derived `aggregator_sync` and
`aggregator_steam_17mafo` excluded from the comparison group so the Steam cash-out wedge
isn't measured against itself — the trailing window sits **~32% ABOVE** them (median
ratio 1.316, a premium on 76.6% of item-days). That is the same Steam cash-out fee wedge
every Steam-derived feed in this archive carries. Excluding these sources therefore pulls
the consensus **down**, not up, unlike the bid.

**The actual defect is the TIME basis, not the level.** A trailing 90-day mean barely
moves when live asks move, so it damps the consensus and mechanically manufactures
mean-reversion in the resulting returns. That matters acutely here because a reversal
effect (`-return_1d` predicting the forward return, rank IC ≈0.17 at 3d) is the signal
this project is currently trying to validate — see
`docs/changelog/2026-08-09-composition-stability-refutes-quoting-artifact.md`. A feed
whose own construction manufactures reversion is not a control for that question; it is a
confound inside it.

This distinction is the same one that separated the bid from these feeds in the first
place: a bid is the wrong side of the book (a level error); a trailing mean is the wrong
time basis (a dynamics error). Both are excluded from voting for that reason, kept as two
separate constants (`BID_SOURCES`, `TRAILING_WINDOW_SOURCES`) rather than merged into one,
so a reader does not read either as "these are bids."

## Measured numbers — corrected

Over 2026, ≥$1, universe-filtered (`archive_universe_sql_filter`: bid, phase-collapsed,
and phantom-slug rows already out):

| | value |
|---|---|
| item-days whose consensus moves | **16.89%** |
| median signed change on moved item-days | **−8.47%** |
| consecutive-day return direction flips | **6.17%** |
| item-days with no other ask source (coverage cost) | **650 of 2,589,787** |

These four numbers were measured by the task's controller and are cited here as the
authoritative figures; the brief this task started from stated different ones (17.13% /
−7.16% / 5.75% / ~670 of 3,093,793) and those are superseded. The 3,093,793 denominator in
the brief could not be reconciled under any reasonable variant of the query — every
attempt landed in the 2.59M–2.71M range — so it should not be cited.

**Independent re-measurement of the coverage cost was attempted and did not complete in
this task.** The full-archive read (13 years, `prices-*.parquet` glob) was too slow to
finish; a second attempt scoped to just the eight 2026 monthly Parquet files
(`prices-2026-01.parquet` … `prices-2026-08.parquet`, 12.7M raw rows before any filter,
read through `db/archive.py::prices_relation` with `archive_universe_sql_filter` applied
— never a raw glob, and no database session opened anywhere in either attempt) was still
running when the task's time budget was spent, and was killed rather than left running
unsupervised. **The 650/2,589,787 figure above is not independently reproduced by this
task; it is the controller's own measurement against the same archive, stated here as
such.**

## Blast radius — corrected

The brief stated "every A/B and every label from 2026-03 onward sits downstream of
this." That is wrong: the three sources exist **only from 2026-07-11 to 2026-08-08** (verified
against the archive — no row before 07-11 or after 08-08 carries any of the three source
names). **Labels and stored A/B verdicts from 2026-07-11 onward** are what sit downstream
of this exclusion, not everything since March. No stored A/B result spanning that window
is citable without re-running it under this change.

## This is NOT a staleness fix

`aggregator_sync` and `aggregator_steam_17mafo` are `last_24h` **falling back** to these
same trailing windows on exactly the illiquid items where `last_24h` has nothing to
report. There is no point-in-time Steam price anywhere in this archive — the trailing
windows are not a secondary vote alongside a clean Steam feed, they are what the "clean"
Steam feed silently becomes on a thin day. Excluding `aggregator_steam_7d/30d/90d` removes
the version of this problem that is visible as a separate source column; it does not (and
cannot, from this archive alone) remove the version hiding inside `aggregator_sync` and
`aggregator_steam_17mafo`'s own fallback logic.

`aggregator_steam_17mafo` (2,169,483 rows, 2026-04-16 → 2026-07-10, flagged unaudited in
`docs/changelog/2026-08-07-bid-source-excluded-from-voting.md`'s "Still open" section) is
**not** excluded here and **must not be**: it is the *only* ask source in the entire
archive for that 86-day span (buff163/csfloat/youpin have a full collection outage across
exactly that window), so excluding it would delete effectively 100% of 2026 price data for
86 days. Its open question — whether it is itself a disguised trailing window rather than
a genuine snapshot — is subsumed by this entry's finding (the mechanism, and the fact that
Steam-side fallback-to-trailing-window is structural, not a one-off) rather than resolved
by it. It remains unaudited.

## The change

`TRAILING_WINDOW_SOURCES = frozenset({"aggregator_steam_7d", "aggregator_steam_30d",
"aggregator_steam_90d"})`, defined in `models/item_parser.py` beside `BID_SOURCES` (both
re-exported by `models/forecaster.py` and exposed as `ItemForecaster` class attributes,
since `_apply_multi_source_voting` — the method that applies them — is itself a
staticmethod called unbound elsewhere in the codebase).

`_apply_multi_source_voting` now drops `BID_SOURCES | TRAILING_WINDOW_SOURCES` in the
same filter, before the `n_ask_sources` count, so a trailing-window leg cannot inflate
the count or the ≥3-source outlier-mask gate. The filter is `frozenset` membership, never
a prefix match: `aggregator_steam_17mafo` is a distinct source name and a
`startswith("aggregator_steam")` filter would have deleted it along with the three
intended sources — catching that was the point of checking it explicitly. `.isin()` on a
`frozenset` stays NULL-safe the same way the bid filter already was: NaN is never `in` the
set, so the whole pre-2026 `source IS NULL` series (9.4M+ rows) keeps voting.

An item-day whose only sources were bids and/or trailing windows now returns no row, via
the same early-return path the bid exclusion already used (with its `n_ask_sources`
`int64` dtype guard preserved — this change makes that branch reachable a new way, and
`test_a_trailing_only_item_day_is_dropped_not_zeroed` covers it).

`VOTED_CACHE_VERSION` **5 → 6.** The workflow's cache key was still at `voted-v4-`
(`.github/workflows/price-forecast.yml`) — the earlier 4→5 bump (for `n_ask_sources`)
never moved this key, so a v5 frame never actually reached CI; the key has been moved
straight to `voted-v6-` in this commit, leaving no dangling v5 key.

## `test_bid_source_voting.py` — why it changed

Not in the brief's file list, and not a weakening of anything in it. That file's `LADDER`
fixture (2026-08-07) uses `aggregator_steam_7d`/`_30d`/`_90d` as three of the ten "generic
ask" legs in a measured 11-source panel, because at the time they voted like any other
ask. This task's exclusion makes those three legs stop voting inside `vote()`
(`ItemForecaster._apply_multi_source_voting`), so two of that file's tests broke on their
expected *values*, not on anything about the bid:

- `test_bid_source_is_excluded_from_the_consensus_median` and
  `test_label_resolution_path_excludes_the_bid` both assert `LADDER_ASK_CONSENSUS`, which
  was `9.045` (the median over ten non-bid legs). With the three trailing-window legs also
  gone, `vote()` on the same fixture now correctly returns `7.32` (median of the seven
  remaining real asks, after the 2σ mask rejects the two `1.000×`-Steam legs that survive
  as `aggregator_sync`/`aggregator_csgotrader`). Verified against the running code, not
  hand-computed. The constant was updated to `7.32` with a comment explaining why; no
  assertion was loosened or removed.
- `test_bid_does_not_count_toward_the_three_source_outlier_gate` built its 3-source
  fixture out of `aggregator_csfloat`, `aggregator_steam_7d` (as a stand-in ask), and the
  bid. Since `aggregator_steam_7d` is now itself excluded, the fixture no longer isolates
  the bid/gate interaction the test is about. Swapped the stand-in to
  `aggregator_youpin` — a real, unaffected ask source — keeping the same multiplier and
  the same assertion (`20.0`, the bare two-ask median). The gate logic under test did not
  change.

`test_published_gate_loader_excludes_the_bid` (also in that file) was unaffected: it
exercises `scripts/walkforward_backtest.py::_load_all_prices`, which never calls
`_apply_multi_source_voting` and only excludes `BID_SOURCES` at the SQL level — this task
did not extend that loader, so its trailing-window legs still vote there. That gap is
noted below.

## What was deliberately not done

- **`scripts/walkforward_backtest.py::_load_all_prices` and the `ab_test_*` private price
  loaders were not touched.** Task 4's brief and the controller's scope were both limited
  to `_apply_multi_source_voting` (the training/prediction/label-resolution consensus).
  The published Backtest Accuracy gate's plain-mean loader — the same loader the bid
  exclusion changelog flagged as needing a separate SQL-level fix — still lets the three
  trailing-window sources vote undiluted. This is a known gap, not an oversight: fixing it
  means extending `archive_universe_sql_filter`/`bid_sources_sql_filter`'s SQL-predicate
  pattern to `TRAILING_WINDOW_SOURCES`, which is a bigger, separately-scoped change (it
  touches every archive reader, not just the training path) and was out of scope here.
- **`aggregator_sync` and `aggregator_steam_17mafo` were not touched**, deliberately — see
  "This is NOT a staleness fix" above. `aggregator_sync` alone covers 2026-01 and 2026-02
  in full for the ≥$1 cohort (52,048 item-days); dropping it to chase the residual
  staleness would cost that entire span for a small further gain.
- **No re-run of any stored A/B or backtest.** This entry records the invalidation of
  everything from 2026-07-11 onward; it does not correct or re-score anything.

## Related

- `docs/changelog/2026-08-07-bid-source-excluded-from-voting.md` — the bid exclusion this
  mirrors in structure, and the origin of the still-open `aggregator_steam_17mafo`
  question this entry subsumes without resolving.
- `docs/changelog/2026-08-09-composition-stability-refutes-quoting-artifact.md` — the
  reversal-signal measurement this exclusion protects the validity of.
- `models/staleness.py` — documents `aggregator_steam_7d/30d/90d` as trailing-window,
  non-point-in-time observations; this entry is what stops them from also voting.
