# Monthly Parquet Partitioning for the Current-Year Price Archive

**Date:** 2026-07-25
**Status:** Approved (design)
**Repos:** `RayanR000/cs2-oracle` (code), `RayanR000/cs2-oracle-data` (archive)

## Problem

The price archive stores one Parquet file per year per dataset
(`prices-YYYY.parquet`, `snapshots-YYYY.parquet`). The daily aggregator does a
read-modify-rewrite of the whole current-year file. Two consequences:

1. **File-size wall.** `prices-2026.parquet` is ~69 MB at 2026-07-25 (Jan–Jul).
   Extrapolated to a full year it reaches ~110 MB, exceeding GitHub's hard
   100 MB-per-file limit — the aggregator's push to `cs2-oracle-data` will start
   failing in the fall. `snapshots-2026.parquet` (~24 MB) grows the same way.
2. **Write cost.** Each daily run rewrites the entire year file.

## Key finding (why this is low-risk)

Every consumer reads the archive via the glob `prices-*.parquet` — either
`read_parquet('.../prices-*.parquet')` in DuckDB or `Path.glob("prices-*.parquet")`
in Python (~25 scripts, `models/forecaster.py`, `collectors/pipeline.py`).
**No reader parses the year out of the filename** (verified by grep). A monthly
file named `prices-YYYY-MM.parquet` still matches `prices-*.parquet`, so readers
need **zero changes**. The forecaster already unions matched files and handles
per-file schema differences, so yearly (old) and monthly (new) files coexist.

A hive layout (`prices/year=YYYY/month=MM/…`) was rejected: it would break every
existing glob.

## Design

### 1. Writer change — `backend/scripts/append_to_parquet.py`

- Compute `month = day_start.month` alongside the existing `year`.
- Change the current-day target filenames:
  - `prices-{year}.parquet` → `prices-{year}-{month:02d}.parquet`
  - `snapshots-{year}.parquet` → `snapshots-{year}-{month:02d}.parquet`
- Applies to **both** write paths in the file (the main path ~line 132 and the
  legacy path ~line 145) and the snapshots path (~line 150).
- Dedup keys are unchanged: `["item_slug", "day", "source"]`. A `day` belongs to
  exactly one month, so per-month dedup is equivalent to per-year dedup.
- `exchange-rates-{year}.parquet` is **unchanged** (8 KB; not size-constrained).

Effect: the daily append now reads/rewrites the current month's file (~10 MB)
instead of the whole year, fixing both the size wall and the write cost.

### 2. One-time migration (executed once, locally)

A migration script/step that:
1. Shallow-clones `cs2-oracle-data`.
2. For `prices-2026.parquet` and `snapshots-2026.parquet`: partition rows by
   `strftime(day, '%Y-%m')` into `prices-2026-01.parquet` … `prices-2026-07.parquet`
   (and snapshots equivalents).
3. **Verifies** before deleting the originals (see Testing).
4. Removes the two yearly 2026 files.
5. Commits and force-pushes `main` (preserving the repo's flat single-commit
   history — orphan commit + `push --force`).

Frozen prior years (2013–2025, each <10 MB) are left untouched.

### 3. Readers

No changes. `prices-*.parquet` continues to match both the frozen yearly files
and the new monthly files.

## Out of scope (YAGNI)

- `exchange-rates-*.parquet` and `ops/*.parquet` (small / separate concern).
- One-off backfill/export scripts (`merge_17mafo_gap.py`, `merge_hf_dataset.py`,
  `export_historical_parquet.py`) — they still write yearly files if re-run,
  which remain glob-compatible and harmless.

## Testing

- **Migration equivalence (blocking, before deletion):** for each split file,
  assert `sum(rows of monthly files) == rows of original yearly file` and that
  the set of `(item_slug, day, source)` keys is identical (zero lost, zero added).
  Mirrors the verification used during the repo migration.
- **Per-month boundary:** assert every row lands in the file matching its
  `day`'s month, and that no monthly file contains rows from another month.
- **Writer unit behavior:** appending a new day writes/updates only that month's
  file and dedups correctly (existing `_append_parquet` semantics).
- **End-to-end:** dispatch the aggregator; confirm it appends to
  `prices-2026-07.parquet` (not a yearly file) and the run is green.
- **Reader smoke:** a `read_parquet('prices-*.parquet')` count over the archive
  equals the pre-migration total (readers see the same data).

## Rollback

The data repo is flat-history and re-derivable; if the split is wrong, restore
from the pre-migration commit / local clone and re-push. The writer change is a
one-line filename edit revert. No reader depends on the layout, so partial
states remain readable.
