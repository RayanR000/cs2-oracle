# 2026-10-09 — The daily archive append merges in DuckDB, written ZSTD and sorted

Performance review 2026-10-08 §4.

## Change

- **`scripts/append_to_parquet.py::_append_parquet` merges in DuckDB.** Until now it read
  the whole month into pandas, concatenated the batch onto it, took the first `ingested_at`
  per key, dropped duplicates and wrote the result back. The month now stays in DuckDB: the
  existing rows with no match in the batch (`NOT EXISTS`), `UNION ALL BY NAME` the batch.
  Semantics are unchanged:
  - A batch row replaces a matching row wholesale.
  - The last duplicate within a batch wins, using an explicit position column.
  - Keys match NULL to NULL (`IS NOT DISTINCT FROM`, the old `dropna=False`).
  - First arrival is kept as `least(new, min(existing))`.
  - A file that predates `source` reads it as NULL.
- **Written ZSTD and sorted on `(source, item_slug, day)`** (`PRICE_KEYS`), not SNAPPY in
  arrival order. No reader depends on row order. Only files this writer rewrites change: the
  current month's file every day, and exchange rates. Older months stay SNAPPY until something
  rewrites them.
- **`collectors/pipeline.py` builds plain dicts, not `PriceHistory` ORM objects.** Nothing
  persisted the ~380k objects; the code flattened them straight back into dicts for the
  snapshot CSV. The review measured them at 2.1 s and +457 MB.

## Measured

The real `prices-2026-09.parquet` (4.54M rows, through 09-19) from the local
`cs2-oracle-data` checkout, plus a 698,502-row batch: the last day's 349,251 rows re-reported
with half the prices changed, and the same rows as a new day. Old and new code each ran on a
separate copy:

| | old (pandas) | new (DuckDB) |
|---|---|---|
| append | 2.67 s | 0.99 s |
| process max RSS (incl. building the batch) | 4.12 GB | 2.09 GB |
| output file | 26.9 MB | 10.5 MB (−61%) |

- **The outputs are identical.** Both hold 5,234,273 rows, `EXCEPT ALL` is empty in both
  directions, the physical Parquet schema is identical (`day` DATE, `ingested_at`
  TIMESTAMP_MICROS), and no key is duplicated.
- **First arrival is preserved.** The re-reported 09-19 rows keep their 09-19 `ingested_at`;
  only the new day carries the batch stamp.
- **The file is much smaller** than the review's −38% across all price files, which spanned the yearly files too.

## Tests

`test_append_to_parquet.py::TestDuckDBMerge` pins the semantics: last-in-batch wins, NULL
keys match each other while first arrival survives, a file without `source` is read as NULL,
and output is ZSTD and key-sorted. The existing 16 append tests pass unchanged. The full suite
passes: 2782 passed, 3 skipped.
