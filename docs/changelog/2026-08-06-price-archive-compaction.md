# Price archive compaction: 114 MB of pure duplication removed

**Date:** 2026-08-06
**Scripts:** `backend/scripts/compact_price_archive.py` (new),
`backend/scripts/append_to_parquet.py`, `backend/scripts/purge_phantom_items.py`,
`.github/workflows/aggregator-update.yml`

> **Production status: not yet applied.** The compaction has run against the local
> `price-archive/` working copy only. The canonical archive is
> `RayanR000/cs2-oracle-data`, still at **210.7 MB** with all 6 snapshot files. To land
> it, dispatch *Aggregator Market Update* with **`compact_archive = true`** — once. See
> "Applying this to production" below.

An audit of the Parquet archive found that 114 MB of its 208 MB was storing the same
numbers twice. Both reclaims are now applied and the writers no longer regenerate them.

## What was redundant

**`median_price`, `min_price`, `max_price` — 69.5 MB.** These were never independent
values. The aggregator's snapshot CSV header is
`item_slug,day,source,price,volume` (`collectors/pipeline.py:276`) — no median column —
so `append_to_parquet.py`'s `median_price=("median_price","mean")` aggregation always
fell through to its `("price","mean")` branch. And because the CSV carries exactly one
row per `(item_slug, day, source)`, the `min`/`max` aggregations returned that single
value. All three were copies of `mean_price` by construction, not by coincidence.

Measured before the change: `median_price == mean_price` in **20,755,907 of 20,755,907
rows**. In `prices-2026-07` the four identical DOUBLEs were **86% of a 56.2 MB file** —
the file that monthly partitioning was introduced to keep under GitHub's 100 MB cap.

No code ever read them. Both hot readers project explicitly:
`forecaster.py:920` and `backtest/price_resolution.py:247` select
`item_slug, day, mean_price, volume, source` and nothing else.

**`snapshots-*.parquet` — 44.5 MB / 10.9M rows.** A strict projection of the prices
file (`SELECT item_slug, day, source, mean_price AS price, volume`), write-only since it
was created. Verified derivable before deleting: every row joined to `prices-*` on
`(item_slug, day, source)` with equal price and volume.

## What was deliberately kept

- **`prices-2026-03` and `prices-2026-04` keep `min_price`/`max_price`.** These two files
  predate the current one-row-per-key CSV and hold **530,111 and 778,786 rows where
  `min != max`** — real intraday range. Dropping them there would have destroyed data.
  The two files lose only `median_price`.
- **`volume`, despite being identically 0 since 2026-05.** `price_resolution.py:249`
  selects it. (That the column is dead is a separate, already-recorded finding — it is
  why the eleven volume features are shelved.)

## The 131 rows that were nearly lost

The safety check refused to delete `snapshots-2026-07`: it held **131 rows with no
counterpart in `prices-2026-07`**, all on **2026-07-11**, the CSV → Parquet cutover day,
across 129 slugs that appear on other days. A partial write on cutover, not junk.

These were absorbed into `prices-2026-07` before the snapshot file was deleted, so the
retirement is lossless. The archive is now 131 rows *larger* than before compaction.
Had the delete been done with `rm snapshots-*.parquet`, those rows would have gone
silently — the check is the reason to run the script rather than the shell.

## Verification

Compared against a full pre-change copy of all 27 files:

| metric | before | after |
|---|---|---|
| rows | 20,755,907 | 20,756,038 (+131 absorbed) |
| distinct item slugs | 41,725 | 41,725 |
| day range | 2013-08-14 → 2026-08-04 | unchanged |
| `sum(volume)` | 13,529,895,902 | unchanged |
| `sum(mean_price)` | 1,104,986,164.547 | +6.790, exactly the 131 orphans' sum |
| `prices-2026-07` | 56.2 MB | 20.9 MB |
| archive Parquet total | 208 MB | 94 MB |

`backend/tests/test_compact_price_archive.py` — 26 tests. Full suite: 895 passed.
Both hot read paths re-run against the compacted archive: 13,718,864 rows with zero null
prices through the forecaster path, 6,898,117 through the resolution path.

**A trap worth recording:** the first `--apply` aborted on its own integrity check.
`sum(mean_price::DOUBLE)` differed in the 13th significant digit between two reads of
identical data — DuckDB aggregates in parallel and float addition is not associative. The
checksum now sums in `DECIMAL`, which is exact and order-independent. A checksum that
isn't reproducible on unchanged data cannot certify a migration; it only produces false
aborts, and would have produced false *passes* had the tolerance been loosened instead.

## Applying this to production

`price-archive/` in this repo is a plain gitignored directory, not a symlink to a
checkout — `AGENTS.md` and `data.md` both claimed otherwise and are corrected in this
change. So compacting it locally reaches nothing. Only `aggregator-update.yml` writes the
data repo, checking it out fresh into `archive/` each run.

`aggregator-update.yml` gained a `compact_archive` boolean dispatch input and a step
gated on it, placed **after** the data-repo checkout and **before** the append, so the
day's rows land on the already-compacted schema rather than being written with NULL
median/min/max and stripped a moment later. Scheduled runs leave `inputs` unset, so the
step is skipped; the script is idempotent, so a stray second run is harmless.

**Do not force-push the local copy over the remote.** Local was at day 2026-08-04 while a
successful aggregator run completed 2026-08-05 23:59 UTC — prod's `prices-2026-08.parquet`
is 12.6 MB against the local 8.4 MB. Pushing local would delete real collection days.

Expected prod result: 210.7 MB → roughly 91 MB (46.8 MB of snapshots plus ~73 MB of
columns, against 162.4 MB of prices files).

Without this step the branch is still safe but pointless in prod: simulated against a
pre-compaction file, the new writer appends cleanly and the file *keeps* all 8 columns,
with `NaN` in the three dead ones for each new row.

`compact_price_archive.py` exits **1** if it finds no `prices-*.parquet` under
`--archive-dir`. Every function treats "no files" as nothing to do, so a typo'd path in
CI would otherwise report success having compacted nothing — the same silent-green failure
mode the zero-row guard in `run_task.py` exists to prevent.

## Not done

- **The one-off merge scripts still emit the old shape.** `merge_hf_dataset.py`,
  `merge_17mafo_gap.py` and `export_historical_parquet.py` still write
  `median_price`/`min_price`/`max_price` and (the first two) `snapshots-*`. None is in a
  workflow and all have already run, so re-running one is a deliberate act — but it would
  re-add both. `compact_price_archive.py` is idempotent and can be re-run to clean up.
- Items 4–8 of the audit are untouched: two divergent Parquet writers (78% of rows have
  no bloom filters and non-pruneable ordering), no working date pruning
  (`day_frac = 1.000` in 19 of 21 files), SNAPPY throughout where ZSTD is free on the
  13 frozen files, `day` typed TIMESTAMP for a midnight-only value, and no Hive
  partitioning. The rewrite machinery in `compact_price_archive.py` makes each of these
  a small follow-up.
- **The largest remaining win is not layout.** The hot 730-day pull scans in 45 ms and
  spends 3,122 ms materialising to pandas. Source fan-out is now 8.78 rows per item-day
  and ~80% of them are discarded by outlier-voted median consensus *after* transfer.
  A pre-voted consensus dataset would cut that materialisation by roughly that factor —
  more than items 4–8 combined.
