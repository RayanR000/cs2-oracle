# `historical_fallback:` re-stamps excluded from `archive_universe_sql_filter`

`collectors/pipeline.py:236-248` re-writes a price up to 7 days old under **today's** `day`
when a collection misses an item, prefixing the source (`historical_fallback:<source>`).
Production's voted path already drops these (`forecaster.py`'s DB read at `:1795` and DuckDB
archive read at `:1919`), but every archive-globbing loader that bypasses the voted path —
`walkforward_backtest.py`, the `ab_test_*` harnesses — did **not**, despite invariant 2
advertising `archive_universe_sql_filter` as carrying all the universe rules. So a stale
re-stamped print entered features and labels under a fresh date in the fresh-model gate and
every A/B. 12,655 rows over exactly six days (2026-07-11..16), on the cohort that failed to
match that day. Deep-model-review §1d — best correctness-per-effort in the review.

## Change

- `models/item_parser.py`: new `historical_fallback_sql_filter()` /
  `HISTORICAL_FALLBACK_PREFIX`, NULL-safe (`source IS NULL OR source NOT LIKE
  'historical_fallback:%'`) like `bid_sources_sql_filter`. Added to
  `archive_universe_sql_filter` under the `source_column` branch, so `source_column=None`
  skips it — safe because the re-stamps are all 2026 feeds, same caveat as the bid rule.
- Tests: `test_ab_harness_universe.py` gains a `historical_fallback` fixture row, a
  drops-only-the-restamp test and a NULL-safety test; the two bid-count assertions shift by
  one; the universe-filter assertion is unchanged (the extra row is an AK-47 the filter drops).

## Not touched

No `VOTED_CACHE_VERSION` bump: the production voted cache already excluded these rows inline,
so its contents are byte-identical. This only changes what the bypassing harnesses and
`walkforward_backtest.py` see.
