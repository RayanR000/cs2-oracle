# 2026-10-09 — The backtest's archive read votes in DuckDB

## Cost

Backtest Accuracy takes 28–35 min a day, the longest step in the daily chain. In run
`37877462648` (day 10-09), 22.8 of its 28.5 minutes were the four `load_voted_prices` calls,
one per `(horizon, model_version)` group, at ~5.7 min each. Everything else (forecast fetch,
frozen-row read, freeze, candidates, verdict refresh, scoring, bias update) took ~5 min.

Locally, on 6,000 slugs over 2026-07-01..09-08, `load_voted_prices` took 38.4 s, and 36.7 s
of it (96%) was the pandas `_apply_multi_source_voting` on 2.4M raw rows. `resolve_anchors`
took 0.4 s for 90K anchors. The forecaster's archive read stopped using the pandas vote on
2026-09-28 (`2026-09-28-voting-moved-into-duckdb.md`); the backtest's loader was left on it.

## Change

- `backtest/price_resolution.py::load_voted_prices` keeps raw rows in DuckDB and votes with
  `ItemForecaster._multi_source_voting_sql`, one query per item-hash chunk (`VOTED_CHUNKS`,
  default 8) under `VOTED_DUCKDB_MEMORY_LIMIT` (default 2 GB), the forecaster's settings.
  `TRY_CAST` plus the NULL/NaN drop replaces `pd.to_numeric(errors="coerce")` + `dropna`.
- The input rows, the window and the signature are unchanged, so every caller
  (`backtest_accuracy`, `candidate_resolution`, `outside_baseline`, the archived replays) gets
  the same frame faster.
- `_apply_multi_source_voting` stays the reference. The new
  `test_backtest_resolution.py::test_sql_vote_matches_the_pandas_reference` pins this loader to
  it over random multi-source item-days at 1 and 3 chunks, and fails when the vote is
  replaced by a plain median.
- No `VOTED_CACHE_VERSION` bump: this loader has no cache.

## Measured (local archive, all 43,227 slugs with a 2026 row, 2026-01-01..09-08)

| Path | Wall time | Voted rows |
|---|---|---|
| old (pandas vote) | 404.8 s | 5,394,253 |
| new (DuckDB vote) | 11.2 s | 5,394,253 |

**Bit-identical:** 0 of 5,394,253 item-day prices differ (max abs diff 0.0), all 5,394,253
anchors resolve identically (5,298,562 on both paths, each to the same `Resolution`), and `stale_run_lookup` agrees on every
key. The 17-row difference quoted in the 2026-10-08 performance review was the old pandas
frame before both implementations took the `VOTE_TIE_RTOL` tie rule; with it, they agree.

## What does not change

- Frozen outcomes: `base_price` / `actual_price` are not re-read, and new resolutions see the
  same voted prices they would have under pandas.
- The per-group read. One vote over the union of groups (review §1) would save ~30 s more at
  ~11 s a pass, and needs `stale_run_lookup` sliced back per group; not worth the diff now.

## Expected

The vote passes fall from ~23 min to well under a minute, putting the daily Backtest at ~6 min.
Read the next scheduled run's `Resolved … anchors` timestamps to confirm.
