---
paths:
  - "backend/db/**"
  - "backend/collectors/**"
  - "backend/scripts/{append_to_parquet,normalize_price_schema}.py"
---

# Reading and writing the price archive

- **Read prices through `db/archive.py::prices_relation`, never a raw glob.** A plain
  `SELECT * FROM read_parquet('prices-*.parquet')` used to return **four** columns and no
  error — DuckDB narrows a multi-file read to the first file's schema and
  `prices-2013.parquet` predates `source`, so `source`/`min_price`/`max_price` silently
  vanished. `scripts/normalize_price_schema.py` gave every file one schema
  (`item_slug, day, source, mean_price, volume, ingested_at`, `day` as `DATE`), but the
  reader still projects an explicit column list and NULLs what is absent, so it works against
  an unmigrated archive too. `source IS NULL` still selects the pre-2026 series — the
  migration materialised a typed NULL rather than stamping a label, because
  `init_local_db.py` derives `is_backfilled` from that equivalence. A materialised column is
  NULLed to `COLUMN_TYPES[c]`, not to VARCHAR: that shortcut was only ever right while
  `source` was the one column that could be missing.
- **`ingested_at` is arrival; `day` is what the row describes.** Added 2026-08-08 and
  **NULL for every row written before it** — the archive had no arrival timestamp anywhere
  and one cannot be reconstructed backwards. A NULL means "arrival unknown" and must never be
  read as "arrived on `day`", which is exactly what 13 years of backfilled rows would falsely
  claim. `append_to_parquet.py` stamps the run's wall clock, not `--date`, so re-exporting an
  old day records an old `day` with a present-day arrival — the truth. **First-arrival
  preservation is a property of that one writer, not of the archive.** `append_to_parquet.py`
  keeps the first arrival explicitly, with a `groupby(dedup_keys)["ingested_at"].transform("min")`
  before its dedup (`:267-270`), so a corrected price does not date the whole month forward.
  **`db/parquet.py::append_monthly` does NOT** — `_append_parquet` selects `_new`
  unconditionally and keeps an existing row only `WHERE NOT EXISTS` a match (`:250-262`), so a
  re-append takes the *new* row's `ingested_at`. Every caller of `append_monthly` that cares
  about arrival has to restore it itself; `collectors/price_history_import.py` does. Corrected
  2026-08-09 — this rule previously stated first-arrival as archive-wide, and one importer was
  written against that claim before it was measured. **The
  migration has not been run**: only CI writes the canonical archive, so the column is absent
  from every stored file until `aggregator-update.yml` runs with `normalize_schema = true`.
- **Operational tables live in `price-archive/ops/*.parquet`.** API routes read Parquet
  first with a DB fallback. See `db/parquet.py`.
- ⚠️ **`ops/forecast_outcomes.parquet` is NOT a mirror of the scored panel, and its gaps are
  BIASED — do not read it for panel work.** Measured 2026-08-12 against prod Postgres: on the
  ≥$1 scored cohort it holds **14,668 of 23,073 rows, missing 8,405 (36%)** across **11 of 22
  (forecast_date, horizon) cells**, and **two whole dates are absent at h=7** (2026-07-31,
  2026-08-04) — dates no published h=7 figure had ever seen. It is **not a lag**: the missing
  2026-07-19 h=14 rows resolved 2026-08-05, six days before the file's own last write.
  **No resolution batch after 2026-08-02 23:12:46 ever landed.** What remains in a deficient
  cell is there only because `_flush_verdict_refresh`
  (`scripts/backtest_accuracy.py`) rewrote it, and that pass writes **only rows whose stored
  verdict differed from the re-derived one** — so the survivors are **selected on
  "the verdict changed"** and their rates are not the population's. 07-19 h=14 reads DA 29.6% /
  down-rate 65.2% on the 115 mirror rows against **39.4% / 51.1%** on the true 1,052.
  **Diagnose a cell before trusting it:** if 100% of its rows have `evaluated_at > resolved_at`
  (in practice sharing one timestamp), it is a verdict-changed subsample. Read prod Postgres
  read-only, or a CI run's `prediction_accuracy`, which reproduced the panel exactly across
  runs `31548564675` / `31557070748`. The file also still carries the 21,737 NULL-`base_price`
  2025-12-01 rows purged from prod — inert only because the ≥$1 filter drops them. This
  corrected nine published figures; see
  `docs/changelog/2026-08-11-the-da-gap-is-the-market-direction-of-five-dates.md`.
- **The ops mirror carries `item_slug`; the DB table does not.** Prices key on the slug
  and ops keys on the Postgres surrogate `item_id`, so the archive could not be joined to
  itself without Supabase. `item_forecasts` and `forecast_outcomes` denormalise the slug
  onto the **Parquet side only** (`_with_item_slug` in `scripts/backtest_accuracy.py`) —
  `to_write` goes to `bulk_insert_mappings`, which errors on an unmapped key. **Every
  mirror writer must supply the column**: `append_table` dedups on the key and replaces
  the whole row, so an omission blanks the slug on everything it touches. That includes
  `_refresh_verdict_columns`, which is why it reads the id→slug map too.
- **Never hand raw dicts to `append_table`.** `db/parquet.py::_jsonify_nested` serialises
  dict/list columns before they reach DuckDB, because DuckDB infers a nested column's SQL
  type *from the batch's contents* — the same code produced `STRUCT`, `MAP(VARCHAR, DOUBLE)`
  and `VARCHAR` for `prediction_accuracy.metrics` on different days, and the frozen on-disk
  struct rejected all but one. Writes go through the serialiser; on the **read** side, run
  `DESCRIBE` rather than assuming a type, because files predating the serialiser still hold
  real Parquet structs. See `tests/test_parquet_nested_columns.py`.
