# 2026-08-07 — One price schema, and an archive that joins to itself

Two defects found in a review of how the data is organised. Both made the store
harder to query than it needed to be, and the first returned wrong answers
rather than errors.

## 1. A plain glob read silently dropped `source`

```sql
DESCRIBE SELECT * FROM read_parquet('prices-*.parquet');
-- item_slug, day, mean_price, volume        <- four columns, no error
```

DuckDB narrows a multi-file read to the schema of the first file, which
alphabetically is `prices-2013.parquet` — and the 13 pre-2026 yearly files
predate the `source` column. Values were never corrupted (checked: Aug 2026 read
through the glob matches the file read directly), but `source`, `min_price` and
`max_price` simply did not exist in the result. A query that filtered or grouped
by source returned a plausible number computed over a column that was not there.

Three call sites had independently grown the same workaround — glob, `DESCRIBE`
each file, emit `NULL::VARCHAR AS source` where missing, `UNION ALL BY NAME`:

- `models/forecaster.py::_fetch_voted_price_history`
- `backtest/price_resolution.py::load_voted_prices`
- `scripts/ab_test_interval_sampling.py::load_features`

Three more (`archive_max_day`, `archive_covered_days`, and the yearly/monthly
`day` type split) hand-rolled per-file unions for the same reason.

**Fixed two ways.** `db/archive.py::prices_relation` is now the one reader: it
projects an explicit column list and NULLs whatever the archive lacks, so it is
correct *before or after* the data migration — which matters, because the
canonical archive is the `cs2-oracle-data` repo and only CI can rewrite it.
`scripts/normalize_price_schema.py` then makes the data itself uniform.

### The migration

Idempotent, dry-run by default, fingerprint-verified per file. Run on the local
copy 2026-08-07 (0.9s wall, all 21 files):

| | before | after |
|---|---|---|
| `SELECT *` columns | 4 | 5 (`item_slug, day, source, mean_price, volume`) |
| `day` type | `TIMESTAMP` (yearly) / `TIMESTAMP_NS` (monthly) | `DATE` everywhere |
| `prices-2026-08` column order | `source` in position 3 | canonical |
| rows | 20,756,038 | 20,756,038 |
| `sum(mean_price)` | 1104986171.33678683 | 1104986171.33678683 |
| `sum(volume)` | 13,529,895,902 | 13,529,895,902 |
| distinct slugs | 41,725 | 41,725 |
| `source IS NULL` rows | 9,429,275 | 9,429,275 |

`source` stays **NULL** on pre-2026 rows deliberately. `init_local_db.py`
derives `is_backfilled` from the equivalence of `source IS NULL` and
`day < '2026-01-01'`; stamping a literal label would break that for no gain.
Materialising the column as NULL makes it addressable while preserving it.

`day` casts to `DATE` losslessly — zero rows across all 21 files carry a time
component, and the script refuses to cast any file that does rather than
truncating silently (a truncation would collapse two observations onto one day
and leave the row count and the sums unchanged, so the fingerprint could not
catch it).

`min_price`/`max_price` survive on `prices-2026-03/04`, which
`compact_price_archive.py` kept because they hold real intraday range. They sit
after the canonical columns.

`append_to_parquet.py` now emits the same shape, so a new month does not drift
back out of it — pinned by
`test_new_month_file_matches_what_the_migration_produces`.

## 2. The archive could not be joined to itself

`price-archive/*.parquet` keys on `item_slug` (the market_hash_name).
`price-archive/ops/*.parquet` keyed only on `item_id`, the Postgres surrogate.
No mapping existed anywhere in the archive, so the most obvious question you can
ask this dataset — *"show me the forecast next to the price history for this
item"* — required a round-trip to production Supabase. That is the network hop
the Parquet store exists to remove.

`item_slug` is now written onto the Parquet mirror of `item_forecasts` and
`forecast_outcomes` at write time, and `scripts/backfill_ops_item_slug.py`
stamps the rows written before that.

**Mirror only.** The column is deliberately *not* added to Postgres — that side
has `items` to join against, and a denormalised copy there would be a second
thing to keep true. `to_write` is handed to `bulk_insert_mappings`, so the two
payloads diverge in `_with_item_slug`.

**Every mirror writer has to carry it.** `append_table` dedups on `forecast_id`
and *replaces the whole row*, so a writer that omitted the column would blank it
on everything it touched. That is why `_refresh_verdict_columns` reads the
mapping too, and why `test_a_second_write_does_not_blank_the_slug` exists.

### Backfill, local copy, 2026-08-07

| Table | Rows | Filled |
|---|---|---|
| `ops/item_forecasts.parquet` | 158,200 | 158,200 |
| `ops/forecast_outcomes.parquet` | 104,642 | 73,220 |
| `ops/event_impacts_denorm.parquet` | 18,473 | 18,473 |

**31,422 `forecast_outcomes` rows have an `item_id` with no `items` row** and
keep a NULL slug. That is a pre-existing referential break between the ops table
and `items` — almost certainly the phantom-item purge — surfaced rather than
introduced by this change. Not addressed here; the script reports it instead of
inventing a value.

The join now works with no database:

```sql
SELECT p.source, count(*) AS n, round(avg(o.direction_correct)*100, 1) AS dir_acc
FROM read_parquet('ops/forecast_outcomes.parquet') o
JOIN read_parquet('prices-*.parquet') p
  ON p.item_slug = o.item_slug AND p.day = CAST(o.target_date AS DATE)
GROUP BY 1;
```

## Production

**Neither migration has run against the canonical archive.** The local
`price-archive/` is an unlinked working copy; only `aggregator-update.yml`
writes `cs2-oracle-data`. Dispatch *Aggregator Market Update* with
`normalize_schema = true` and `backfill_ops_slug = true` — both idempotent, both
safe to leave off afterwards, both ordered before the daily append.

The code change is safe to ship ahead of that: `prices_relation` reads a
migrated and an unmigrated archive identically.

## Tests

`tests/test_archive_reader.py` (17), `tests/test_normalize_price_schema.py`
(21), `tests/test_ops_item_slug.py` (18), plus 5 added to
`tests/test_append_to_parquet.py`. Full suite 1,224 passed.

`test_plain_glob_silently_drops_source` asserts the DuckDB behaviour that made
this necessary — if a future version stops narrowing, that test fails and the
reader can be simplified.
