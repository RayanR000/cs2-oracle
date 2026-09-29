# Archive day 2026-09-19: lost to a publish race, recoverable, and the race closed

**Date:** 2026-09-29

## What happened

Day 2026-09-19 was **collected and published, then deleted**:

| UTC (09-20) | data-repo push | commit |
|---|---|---|
| 00:57:54 | Aggregator run 35480028849 | `940531b → 284f9d9` "archive: prices + exchange-rates 2026-09-19" |
| 00:58:27 | an `item_forecasts 2026-09-20` publish | `284f9d9 → 561e007` |

The Aggregator resolved the day correctly and appended 349,187 price rows. The next day's run
started from a file without them: 5,234,209 rows before its append, 5,234,470 after an append of
349,448. The overwriting commit came from a publish whose tree was checked out before 00:57:54.
The run ID the 2026-09-23 changelog cites for that serve (35480138859) falls between the
Aggregator's and the chained Backtest's, and now returns 404. So it was most likely a manual
Price Forecast dispatch that overlapped the Aggregator.

Every publisher rebuilds the data repo as an orphan commit and runs `git push --force`, so a stale
checkout deletes whatever landed after it. The comment in `price-forecast.yml` assumed a
sequential chain. Manual dispatches are not in the chain, and the workflows use separate
concurrency groups.

The overwrite hit four files. The tree diff between `284f9d9` and `561e007` differs only in
these:

| file | 09-19 rows at `284f9d9` | on origin 2026-09-28 |
|---|---|---|
| `prices-2026-09.parquet` | 349,187 (39,337 items, 12 sources) | 0 |
| `supply-2026-09.parquet` (`snapshot_day`) | 99,099 | 0 |
| `volume-2026-09.parquet` | 36,250 | 0 |
| `exchange-rates-2026.parquet` | 51 | 0 |

The csgotrader source is `latest/*.json`, which is live only, so the day cannot be re-collected.
GitHub still serves `284f9d9` by SHA, and that copy is the recovery source.

## The fix

1. **`scripts/restore_archive_day.py`** appends the source's rows for one day, only to files that
   hold none for it. It keeps each file's schema and compression, and keeps the original
   `ingested_at` / `collected_at`, since the rows first arrived on 2026-09-20. The write is
   atomic, and a rerun is a no-op. 7 tests.
2. **`archive-restore-day.yml`** is a `workflow_dispatch` job (dry run unless `apply`) that runs it
   in CI. It keeps "only CI writes the data repo" true.
3. **Every publisher now pushes with `--force-with-lease=main:$BASE`**, where `$BASE` is the SHA it
   checked out: Aggregator, Price Forecast, Backtest Accuracy, Event Correlation and the restore
   job. A stale checkout now fails its push instead of deleting newer data.
   `tests/test_data_repo_publish_lease.py` fails on any bare `git push --force` in a workflow
   that does an orphan checkout.
   - **Cost:** an overlapping manual dispatch now turns red at publish, and its Parquet mirror
     write is lost (the DB write already happened). Re-dispatch it.

## Rehearsed on the real files

Applied to copies of origin (`21180a1`) from `284f9d9`: all four files took the day, and a second
run skipped all four. Read back through `prices_relation` with `archive_universe_sql_filter`,
09-17..21 hold 320,757 / 320,877 / **320,805** / 320,963 / 321,061 rows. There are 0 duplicate
`(item_slug, day, source)` keys, and every other day matches origin exactly (0 rows added,
0 removed).

## To run after merge

Dispatch **Archive Restore Day** with `day=2026-09-19`,
`from_ref=284f9d9566bfd08da70b080b53c6d156feb0859a` and the default files. Run it once with
`apply` unticked, read the report, then run again with `apply`. Dispatch outside 22:30–03:30 UTC;
the lease makes an overlap fail safely, but it is still wasted work.

Once 09-19 is back, h=3 forecast dates 09-16/17/18 become scoreable (next-steps item 2),
before the PID read's window closes (10-18) and its single read (~10-23).
