# Data Architecture

## Motivation

Supabase has a 500 MB limit and the daily OHLCV history does not fit in it. Analysis and
training need full daily granularity for lags, SMAs, momentum, event impact and forecasts.

**Solution:** full history lives in Parquet, in a separate repo. Supabase is a lean
serving layer and a fallback. Training and analysis read Parquet locally via DuckDB
instead of querying Supabase over the network.

---

## Current Architecture

The canonical archive is the repo **`RayanR000/cs2-oracle-data`**, branch `main`,
force-pushed as a squashed orphan commit by every workflow that writes it. There is no
`data-archive` branch — any command referencing one is dead.

Locally, `price-archive/` is a gitignored **plain directory** (`.gitignore:65`) — despite
what `.gitignore:64` says, it is not a symlink and not a checkout. It is an unlinked local
working copy, and it lags the canonical repo. **Compacting or repairing it does not reach
production**; only `aggregator-update.yml` writes the data repo, which it checks out fresh
into `archive/` each run. The empty `../cs2-oracle-data` checkout has no commits and is
not wired to anything. Never force-push the local copy over the remote — it is behind.

```
price-archive/                       (local working copy, NOT the canonical repo)
  ├─ prices-YYYY.parquet             — item_slug, day, mean_price, volume (yearly, pre-2026)
  ├─ prices-YYYY-MM.parquet          — + source (monthly from 2026 on)
  ├─ exchange-rates-YYYY.parquet     — currency rates
  ├─ player-counts-YYYY.parquet      — frozen; the collector was removed in 181488b
  ├─ item-metadata.parquet           — 8,691 item rows
  └─ ops/                            — operational tables, one Parquet file per table
       accuracy_alerts, collection_runs, event_impacts_denorm, events,
       forecast_outcomes, item_forecasts, prediction_accuracy, supply_snapshots

Supabase (serving + fallback):
  ├─ items (+ is_backfilled)         — the only thing training reads from the DB
  ├─ price_history                   — stale; the aggregator writes to Parquet only
  ├─ events / event_impacts / event_correlations
  ├─ collection_runs                 — run tracking
  ├─ item_forecasts / prediction_accuracy / forecast_outcomes / accuracy_alerts
  ├─ supply_snapshots                — frozen, collector deleted
  ├─ social_mentions                 — 0 rows all-time, collector deleted
  └─ users
```

**`ops/` is read before the DB.** `db/parquet.py:37-48` points at `price-archive/ops/` and
API routes query it first, falling back to Supabase only when the Parquet read returns
nothing or raises — see `api/routes/items.py:418-424` for the pattern, repeated for
trends, predictions, item events, event impacts and sentiment. Nested values in `ops/` are
stored as **JSON text store-wide** (`db/parquet.py::_jsonify_nested`), because DuckDB
infers a nested column's SQL type from the batch's contents; see
`docs/changelog/2026-08-05-backtest-red-triage.md`.

Prices and snapshots are partitioned **monthly** from 2026 onward
(`scripts/append_to_parquet.py:5-9`) to stay under GitHub's 100 MB per-file limit;
`prices-2026-07.parquet` alone is 56 MB. The loader globs `prices-*.parquet`, so a
restored single-file `prices-2026.parquet` would be read *alongside* the monthly set.

### Data Flow

```
Daily aggregator ──▶ snapshot CSV ──▶ append_to_parquet.py ──▶ prices-YYYY-MM.parquet
                                                               exchange-rates-YYYY.parquet

Training / backtest / analysis (DuckDB + read_parquet over price-archive/)
  └─ fetch_price_history(backfilled_only=True) — local, no network
  └─ results ──▶ price-archive/ops/*.parquet  (+ Supabase mirror)

API serving:
  GET /items/{id}/price-history   → Supabase price_history only  (see below)
  GET /items/{id}/trends          → ops/item_forecasts.parquet, DB fallback
  GET /items/{id}/prediction      → ops/item_forecasts.parquet, DB fallback
  GET /items/{id}/events|impacts  → ops/*.parquet, DB fallback
```

**Long-range price history does not work.** `api/routes/items.py:234-257` accepts
`days` up to 5000 and queries Supabase `PriceHistory` unconditionally — there is no
DuckDB/Parquet branch at any `days` threshold. Because `price_history` is stale, any
request beyond the last few days of coverage returns near-nothing, silently. Routing this
endpoint at the archive the way the forecast and event endpoints already do is unbuilt
work, not shipped behaviour.

### Storage Breakdown

Measured 2026-08-06 against the **local** copy, after the compaction described in
`docs/changelog/2026-08-06-price-archive-compaction.md`. The canonical repo is still
uncompacted at **210.7 MB** until *Aggregator Market Update* is dispatched once with
`compact_archive = true`; expect ~91 MB after that. Prod also holds one or two more days
than the local copy, so its per-file sizes run slightly higher.

| File / group | Size | Rows |
|--------------|------|------|
| `prices-2026-*.parquet` (8 monthly files, Jan–Aug) | 55 MB | **11,326,763** |
| `prices-20XX.parquet` (pre-2026, yearly) | 33 MB | 9,429,275 |
| `exchange-rates-2026.parquet` | 5 KB | 306 (6 distinct days, latest 2026-07-17) |
| `item-metadata.parquet` | 0.1 MB | 8,691 |
| `ops/item_forecasts.parquet` | 1.6 MB | 158,200 |
| `ops/forecast_outcomes.parquet` | 2.2 MB | 104,642 |
| `ops/event_impacts_denorm.parquet` | 0.7 MB | 18,473 |
| `ops/supply_snapshots.parquet` | 0.5 MB | 35,037 (frozen) |
| `ops/collection_runs.parquet` | <0.1 MB | 194 |
| `ops/prediction_accuracy.parquet` | <0.1 MB | 84 |
| `ops/events.parquet` | <0.1 MB | 79 |
| `ops/accuracy_alerts.parquet` | <0.1 MB | 13 |

The price archive spans **2013-08-14 → 2026-08-04** and carries **41,725 distinct item
slugs**. The local item catalog is separate: `backend/runtime/market_catalog.db`, 18 MB,
31,908 `market_items`.

Growth is dominated by the daily append: **~362,586 OHLCV rows/day** across 11 source
labels. `ops/` tables are UPSERT-or-append-and-dedup and stay under a few MB each;
`forecast_outcomes` is insert-only (see below).

### Performance

| Operation | Before | After |
|-----------|--------|-------|
| Training / analysis read | Supabase query over network (~2-5s) | DuckDB local Parquet (~200ms) |
| API listing filter | Correlated `EXISTS` subquery on `price_history` | `is_backfilled` column index |
| API forecast / event reads | Supabase round-trip | `ops/*.parquet` via DuckDB, DB fallback |

---

## Schema Changes

### `items` table — `is_backfilled` column

```python
is_backfilled = Column(Integer, default=0)  # boolean: has CSMarketAPI historical series
```

Marks the items carrying the CSMarketAPI historical series — **not** merely "present in the
archive". It is **derived from the archive, not set by hand**: `scripts/init_local_db.py`
selects the slugs with rows before 2026-01-01 (that series predates the `source` column, so
the same set is what `source IS NULL` selects) and re-derives the flag on **every run**,
correcting rows written by older versions (`init_local_db.py:66-134`). ~5,542 items are
flagged.

This replaced a static blanket flag that read 100% of items, which made
`backfilled_only=True` admit the low-history live cohort while excluding most of the grown
archive — it mis-scoped both training and predict. See commit `065613b`.

`init_local_db.py` refuses to run against a non-local database (`assert_local_db()`),
because `backend/.env` points `DATABASE_URL` at production Supabase and the engine binds at
import time.

### `price_history` composite PK

`(item_id, timestamp, source)` promoted to primary key. Dropped surrogate `id` bigint PK
(no FKs referenced it). Freed ~80 MB index space.

### `backfilled_item_clause()` rewritten

Before: `EXISTS (SELECT 1 FROM price_history WHERE item_id=Item.id AND source IN ('market_csgo','steam_historical'))`

After: `Item.is_backfilled == 1` (`database.py:105`)

### Migration summary

| Migration | What it does |
|-----------|-------------|
| 0001 | Initial schema: items, price_history, daily_analysis, events |
| 0002 | Expand price_history source column, add supply_snapshots |
| 0003 | Add unique constraint on price_history, item_forecasts table |
| 0004 | Add item metadata images columns |
| 0005 | Add performance indexes |
| 0006 | Composite PK on price_history |
| 0007 | Add `is_backfilled` + create `chart_points` (later dropped) |
| 0008 | Drop redundant chart_point index, clean stale price_history rows |
| 0009-0010 | Prune and drop `trend_indicators` table |
| 0011 | Add `prediction_accuracy` table |
| 0012 | Drop `chart_points` table (data lives in Parquet) |
| 0013 | Add `accuracy_alerts` table |
| 0014 | Add `forecast_outcomes` table |
| 0015 | **Drop `daily_analysis` table** (data in Parquet + item_forecasts) |
| 0016 | Add `supply_snapshots` table |
| 0017 | Add item rarity columns |
| 0018 | Add `social_mentions` table — Reddit sentiment (VADER) |
| 0019 | Add `base_price` / `resolved_at` to `forecast_outcomes`, `price_tier` to `prediction_accuracy`; relax `current_price` to nullable |
| 0020 | Include `price_tier` in the `prediction_accuracy` unique constraint |

> `alembic upgrade` cannot replay this chain from scratch on SQLite — revisions
> `0001`→`0018` contain Postgres-only `ALTER COLUMN ... TYPE` DDL. To rehearse
> against a SQLite snapshot, `alembic stamp 0018` first; `0019`/`0020` are
> dialect-aware and apply to both.

**`forecast_outcomes` is insert-only.** Once a row has `base_price`,
`actual_price` and `resolved_at`, those three columns are final — the daily
backtest is structurally incapable of moving them, so a green run cannot silently
restate history. `--rescore` recomputes the derived verdict columns from the
frozen actuals without reading the archive; `--reresolve` is the only path that
re-reads the archive, overwrites frozen actuals, and deletes rows whose forecast
no longer resolves. See `docs/changelog/2026-08-01-deterministic-backtest.md`.

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `append_to_parquet.py` | Daily: snapshot CSV → monthly `prices-YYYY-MM` + yearly `exchange-rates-YYYY` |
| `compact_price_archive.py` | One-off: drop the redundant `median_price`/`min_price`/`max_price` columns and retire `snapshots-*`. Idempotent; dry-run by default |
| `db/parquet.py` | The `ops/` store: `append()` (concat-and-dedup, full rewrite) and `query()` (DuckDB context manager). Serialises nested values to JSON text |
| `init_local_db.py` | Rebuild a local `items` table from the archive and re-derive `is_backfilled` |
| `export_historical_parquet.py` | One-time: csmarketapi.db → year-split Parquet |
| `merge_hf_dataset.py` | One-time: HuggingFace CS2 dataset → 2026 Parquet |

### Daily run

```
Aggregator Market Update — GitHub Actions, cron 23:00 UTC
  ├─ run_task.py migrate
  ├─ Resolve snapshot date  (collectors/snapshot_date.py → AGGREGATOR_SNAPSHOT_DATE)
  ├─ run_task.py aggregate
  │    ├─ 7 CSGOTrader price endpoints + exchange_rates.json
  │    ├─ 11 source labels → /tmp/aggregator-snapshots-$DATE.csv
  │    ├─ /tmp/aggregator-backfilled-$DATE.csv, /tmp/exchange-rates-$DATE.csv
  │    └─ CollectionRun row (no prices written to Supabase)
  ├─ Checkout RayanR000/cs2-oracle-data  (needs CS2_DATA_REPO_TOKEN)
  ├─ append_to_parquet.py --date $SNAPSHOT_DATE --out-dir ../archive
  │    └─ collapse to daily OHLCV → prices-YYYY-MM / exchange-rates-YYYY
  └─ Publish updated archive: orphan commit + force-push to main

  ▼ workflow_run
Price Forecast  ──▶  ops/item_forecasts.parquet
  ▼ workflow_run
Backtest Accuracy  ──▶  ops/forecast_outcomes.parquet, ops/prediction_accuracy.parquet
```

The snapshot date is resolved **once** and pinned for every later step. Reading the clock
twice silently lost 2026-07-27, 07-30 and 08-03 from the archive
(`aggregator-update.yml:56-68`).

---

## Key Schema Milestones

See `docs/changelog/` for full detail. Major changes:
- **2026-07-07**: Composite PK on `price_history` promoted (`item_id, timestamp, source`), freed ~80 MB
- **2026-07-08**: Backfilled catalog, 1×/day collection, Parquet archive introduced
- **2026-07-08**: Dropped `trend_indicators` table, cleared stale rows (~350 MB recovered)
- **2026-07-11**: All sources written to Parquet; dropped `chart_points` (freed 290 MB)
- **2026-07-16**: Dropped `daily_analysis` table (migration 0015)
- **2026-07-20**: Merged HF CS2 dataset into the Parquet archive (`2026-07-20-hf-dataset-merge.md`)
- **2026-08-02**: Archive moved to the `cs2-oracle-data` repo; 2026 prices split into monthly files after a missing day invalidated a 5,542-forecast backtest cohort
- **2026-08-05**: `is_backfilled` re-derived from the archive on every run (`065613b`); `ops/` nested values standardised on JSON text (`2026-08-05-backtest-red-triage.md`)
