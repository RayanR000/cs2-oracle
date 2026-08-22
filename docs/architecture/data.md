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

Locally, `price-archive/` is a gitignored **plain directory** (`.gitignore:71`) — despite
what `.gitignore:70` says, it is not a symlink and not a checkout. It is an unlinked local
working copy, and it lags the canonical repo. **Compacting or repairing it does not reach
production**; only `aggregator-update.yml` writes the data repo, which it checks out fresh
into `archive/` each run. The empty `../cs2-oracle-data` checkout has no commits and is
not wired to anything. Never force-push the local copy over the remote — it is behind.

```
price-archive/                       (local working copy, NOT the canonical repo)
  ├─ prices-YYYY.parquet             — item_slug, day, source, mean_price, volume,
  │                                    ingested_at (yearly, pre-2026; source is NULL
  │                                    there, ingested_at NULL before 2026-08-08)
  ├─ prices-YYYY-MM.parquet          — same six columns (monthly from 2026 on)
  │                                    2026-03/04 also carry min_price, max_price
  ├─ exchange-rates-YYYY.parquet     — currency rates
  ├─ player-counts-YYYY.parquet      — frozen; the collector was removed in 181488b
  ├─ item-metadata.parquet           — 8,691 item rows
  ├─ supply-YYYY-MM.parquet          — item_slug, snapshot_day, source, listing_count,
  │                                    ask-ladder quantiles, depth, listing age (monthly)
  └─ ops/                            — operational tables, one Parquet file per table
       accuracy_alerts, collection_runs, event_impacts_denorm, events,
       forecast_outcomes, item_forecasts, prediction_accuracy

Supabase (serving + fallback):
  ├─ items (+ is_backfilled, is_trainable) — `is_backfilled` is the SERVE universe and the
  │                                    only thing predict reads from the DB; the TRAIN
  │                                    universe is derived from the archive instead
  │                                    (`_resolve_backfilled_slugs`, forecaster.py:1953)
  ├─ price_history                   — stale; the aggregator writes to Parquet only
  ├─ events / event_impacts / event_correlations
  ├─ collection_runs                 — run tracking
  ├─ item_forecasts / prediction_accuracy / forecast_outcomes / accuracy_alerts
  ├─ supply_snapshots                — deprecated; one stale day (2026-07-15, 35,037 rows),
  │                                     collector deleted. Not published to the data repo;
  │                                     the local Parquet was deleted 2026-08-06. Any revival
  │                                     writes a new slug-keyed series, not this table.
  ├─ social_mentions                 — 0 rows all-time, collector deleted
  └─ users
```

**`ops/` rows carry `item_slug` as well as `item_id`.** The prices files key on
the slug and the ops tables key on the Postgres surrogate, so until 2026-08-07
joining a forecast to its own price history needed a round-trip to Supabase.
`item_forecasts` and `forecast_outcomes` now denormalise the slug onto the
**Parquet mirror only** — the DB side has `items` to join against. `append_table`
replaces whole rows on the dedup key, so *every* mirror writer must supply the
column or it blanks. `scripts/backfill_ops_item_slug.py` fills pre-existing rows;
31,422 `forecast_outcomes` rows point at `item_id`s with no `items` row and keep
a NULL slug.

**`ops/` is read before the DB.** `db/parquet.py:37-38` points at `price-archive/ops/` and
API routes query it first, falling back to Supabase only when the Parquet read returns
nothing or raises — see `api/routes/items.py:555-562` for the pattern (`_trends_parquet`
then the DB query), repeated for predictions (:711), item events (:838), event impacts
(:864) and sentiment (:1041). Nested values in `ops/` are
stored as **JSON text store-wide** (`db/parquet.py::_jsonify_nested`), because DuckDB
infers a nested column's SQL type from the batch's contents; see
`docs/changelog/2026-08-05-backtest-red-triage.md`.

Prices and snapshots are partitioned **monthly** from 2026 onward
(`scripts/append_to_parquet.py:5-9`) to stay under GitHub's 100 MB per-file limit;
`prices-2026-07.parquet` alone is 56 MB. The loader globs `prices-*.parquet`, so a
restored single-file `prices-2026.parquet` would be read *alongside* the monthly set.

### Reading the price archive

**Go through `db/archive.py::prices_relation`, not a raw glob.** Every prices
file now shares one schema (`CANONICAL_PRICE_COLUMNS` =
`item_slug, day, source, mean_price, volume, ingested_at`, `day` as `DATE`)
after `scripts/normalize_price_schema.py`, but the reader still
projects an explicit column list and NULLs what is absent, so it is correct
against an unmigrated archive too — which is what a fresh clone of the data repo
is until *Aggregator Market Update* is dispatched with `normalize_schema = true`.

Before that migration a plain `SELECT * FROM read_parquet('prices-*.parquet')`
returned **four** columns and no error: DuckDB narrows a multi-file read to the
first file's schema, and `prices-2013.parquet` predates `source`. Values were
correct; `source` silently did not exist. Three call sites had each grown the
same per-file `DESCRIBE` + `NULL AS source` workaround. See
`../changelog/2026-08-07-archive-schema-and-keys.md`.

`source IS NULL` still means "the pre-2026 CSMarketAPI series" — the migration
materialised the column as a typed NULL rather than stamping a label, precisely
so the `is_backfilled` derivation below keeps working.

**The item universe is a read-time filter, not a property of the archive.** Every
row stays on disk; readers narrow it. Two rules apply at the SQL/archive-glob
level, and a new reader has to carry both itself — `walkforward_backtest.py`
needed a separate copy of each, because it globs the archive rather than
calling `fetch_price_history`: `models/item_parser.py::BID_SOURCES`
(`aggregator_buff163_buy` is a bid and must not vote), and
`models/item_parser.py::phase_collapsed_sql_filter()`, which drops the Doppler
and Gamma Doppler names because one such name prices every phase at once. The
phase rule matches **129 of 41,725 slugs / 47,081 of 20,756,038 rows**
(0.309% / 0.227%), of which two are false positives —
`Sticker | Doppler Poison Frog (Foil)` and its Sticker Slab twin, 396 rows, kept
by `PHASE_COLLAPSED_EXEMPT_PATTERNS`. Both filters are written NULL-safe: a bare
`NOT LIKE` over a NULL evaluates to NULL and drops the row, and NULL selects 13
years here.

Two further rules, `models/item_parser.py::TRAILING_WINDOW_SOURCES`
(`aggregator_steam_7d/30d/90d`, trailing-window mean sale prices that must not
vote either, as of 2026-08-09) and `::STEAM_SPOT_SOURCES`
(`aggregator_steam_spot`, Steam's fallback-free `last_24h`, which would cast a
second Steam ballot beside `aggregator_sync`, as of 2026-08-17), apply only inside
`ItemForecaster._apply_multi_source_voting` (forecaster.py:2103) — they are not part of
`archive_universe_sql_filter()`, so they constrain anything that routes through
the vote (`fetch_price_history`), not a raw archive glob. `walkforward_backtest.py`
and the `ab_test_*` harnesses glob the archive and apply
`archive_universe_sql_filter()` directly, so they do not exclude these four
sources from their own averages — a known, deferred gap (see
`.claude/rules/item-universe.md` and `.claude/rules/labels-and-embargo.md`).
See `../changelog/2026-08-08-phase-collapsed-names-dropped.md` and
`../changelog/2026-08-07-bid-source-excluded-from-voting.md`.

### Data Flow

```
Daily aggregator ──▶ snapshot CSV ──▶ append_to_parquet.py ──▶ prices-YYYY-MM.parquet
                                                               exchange-rates-YYYY.parquet
                 └──▶ run_supply_depth.py ─────────────────▶ supply-YYYY-MM.parquet
                      (same workflow, between append and publish, so the day's
                       supply rows land in the same orphan commit as its prices)

Training / backtest / analysis (DuckDB + read_parquet over price-archive/)
  └─ fetch_price_history(backfilled_only=True) — local, no network
  └─ results ──▶ price-archive/ops/*.parquet  (+ Supabase mirror)

API serving:
  GET /items/{id}/price-history   → Supabase price_history only  (see below)
  GET /items/{id}/trends          → ops/item_forecasts.parquet, DB fallback
  GET /items/{id}/prediction      → ops/item_forecasts.parquet, DB fallback
  GET /items/{id}/events|impacts  → ops/*.parquet, DB fallback
```

**Long-range price history does not work.** `api/routes/items.py:376-397` accepts
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
| `supply-2026-08.parquet` | 1.6 MB | 84,408 (1 day, 30,330 items, 4 feeds) |
| `ops/item_forecasts.parquet` | 1.6 MB | 158,200 |
| `ops/forecast_outcomes.parquet` | 2.2 MB | 104,642 (**local copy** 114,489 / **durable** 70,409 as of 2026-08-12 — see the warning below) |
| `ops/event_impacts_denorm.parquet` | 0.7 MB | 18,473 |
| `ops/collection_runs.parquet` | <0.1 MB | 194 |
| `ops/prediction_accuracy.parquet` | <0.1 MB | 84 |
| `ops/events.parquet` | <0.1 MB | 79 |
| `ops/accuracy_alerts.parquet` | <0.1 MB | 13 |

The price archive spans **2013-08-14 → 2026-08-04** and carries **41,725 distinct item
slugs** over **20,756,038 rows**. The local item catalog is separate:
`backend/runtime/market_catalog.db`, 18 MB, 31,908 `market_items`. What readers *use*
is narrower than what is stored — see § Reading the price archive.

For what that coverage actually amounts to — history depth per item, per-source spans and
gaps, which columns still carry information, and label coverage — see
`../references/data-inventory.md`. The short version: only 5,542 of the 41,725 slugs have
more than 180 days of history, and over the last 90 days `mean_price` is the only
non-degenerate column.

Growth is dominated by the daily append: **~362,586 OHLCV rows/day** (measured 2026-08-01)
across the then-11 source labels; `aggregator_steam_spot` was added 2026-08-17 and is not in
that figure. `ops/` tables are UPSERT-or-append-and-dedup and stay under a few MB each;
`forecast_outcomes` is insert-only (see below).

> ⚠️ **`ops/forecast_outcomes.parquet` exists in two copies and neither is the full scored panel.
> Query prod Postgres read-only for panel work.** All three stores were measured 2026-08-12, the
> durable one by fetching its own blob:
>
> | store | rows | dates | freshness | `evaluated_at > resolved_at` |
> |---|---|---|---|---|
> | prod Postgres — complete | **121,699** | **12** | max `resolved_at` 2026-08-12 00:05:52.489046 | — |
> | **durable** (`RayanR000/cs2-oracle-data`, head `c2b96cad04ce`, run `31557070748`) | 70,409 | 10 | matches Postgres to the microsecond | 57 / 70,409 (**0.1%**) |
> | **local working copy** (`price-archive/`, gitignored) | 114,489 | 13 | max `resolved_at` 2026-08-11 00:19:05, max `evaluated_at` 2026-08-11 21:10:28 | **100%** in every deficient cell |
>
> **The publish leg works — do not go looking for a dead one.** Every cell the local copy is missing
> is complete in the durable file (07-19 h=14 is 1,052 rows there against 115 locally). What the
> durable file is instead is **shallow**: 10 dates, with 2025-12-01 and 2026-07-17 absent entirely,
> so it cannot answer a history question. Why it is 10 dates and not 12 is **untraced** — it is fresh
> and internally complete, so this is not staleness.
>
> **The local copy is the trap**, and it is what `db/parquet.py` reads on a developer machine. It
> went stale for resolution batches after **2026-08-02 23:12:46**, then a verdict-refresh run against
> prod on 2026-08-11 21:10 wrote rows back into it; `_flush_verdict_refresh` writes **only rows whose
> stored verdict differed**, so each deficient cell's survivors are selected on "the verdict changed"
> and their rates are not the population's. On the ≥$1 scored cohort it holds **14,668 of 23,073 rows
> — missing 36%** across 11 of 22 (forecast_date, horizon) cells, with two whole dates absent at h=7.
> It also still carries the **21,737 NULL-`base_price` 2025-12-01 rows purged from prod**.
>
> **Diagnose a cell before trusting either copy:** if ~100% of its rows have
> `evaluated_at > resolved_at`, sharing one timestamp, it is a verdict-changed subsample. The durable
> file reads 0.1% by that test, the local copy's deficient cells 100%. A CI run's
> `prediction_accuracy` is also sound.
> `../changelog/2026-08-11-the-da-gap-is-the-market-direction-of-five-dates.md`.

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
correcting rows written by older versions (`init_local_db.py:72-148`). ~5,542 items are
flagged (as of 2026-08-06). The same run derives `is_trainable` (migration 0023), which
narrows `is_backfilled` by dropping iflow-only history — but training does **not** read that
column: `_resolve_backfilled_slugs(universe="train")` re-derives the train cohort from the
archive, because a DB read of a column prod did not have fell back to loading all 41,885
slugs and OOMed a cold retrain.

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
| 0021 | Add `base_stale_run_days` to `forecast_outcomes` |
| 0022 | Add `anchor_clean` / `anchor_wedge_pct` to `item_forecasts` |
| 0023 | Add `is_trainable` to `items` — the ORM column that shipped with no migration and crashed prod's aggregator on `SELECT items.is_trainable` |
| 0024 | Add `exceed_p` to `item_forecasts` — the served exceedance probability |

> `alembic upgrade` cannot replay this chain from scratch on SQLite — revisions
> `0001`→`0018` contain Postgres-only `ALTER COLUMN ... TYPE` DDL. To rehearse
> against a SQLite snapshot, `alembic stamp 0018` first; `0019`/`0020` are
> dialect-aware and apply to both, as are `0021`–`0024`.

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
| `db/archive.py` | The one price-archive reader: `prices_relation()` (typed projection over the glob), `price_files()`, `canonical_order()` |
| `append_to_parquet.py` | Daily: snapshot CSV → monthly `prices-YYYY-MM` + yearly `exchange-rates-YYYY`, in canonical order with `day` as DATE |
| `normalize_price_schema.py` | One-off: one schema for every prices file — materialise `source`, cast `day` to DATE, canonical order. Idempotent; dry-run by default |
| `backfill_ops_item_slug.py` | One-off: stamp `item_slug` onto existing `ops/*.parquet` rows. Idempotent; dry-run by default |
| `compact_price_archive.py` | One-off: drop the redundant `median_price`/`min_price`/`max_price` columns and retire `snapshots-*`. Idempotent; dry-run by default |
| `db/parquet.py` | The `ops/` store: `append()` (concat-and-dedup, full rewrite) and `query()` (DuckDB context manager). Serialises nested values to JSON text |
| `init_local_db.py` | Rebuild a local `items` table from the archive and re-derive `is_backfilled` |
| ~~`export_historical_parquet.py`~~ | One-time: csmarketapi.db → year-split Parquet. **Script deleted**; only the changelog record survives |
| ~~`merge_hf_dataset.py`~~ | One-time: HuggingFace CS2 dataset → 2026 Parquet. **Script deleted**; see `docs/changelog/2026-07-20-hf-dataset-merge.md` |

### Daily run

```
Aggregator Market Update — GitHub Actions, cron 23:00 UTC
  ├─ run_task.py migrate
  ├─ Resolve snapshot date  (collectors/snapshot_date.py → AGGREGATOR_SNAPSHOT_DATE)
  ├─ run_task.py aggregate
  │    ├─ 7 CSGOTrader price endpoints + exchange_rates.json
  │    ├─ 12 source labels → /tmp/aggregator-snapshots-$DATE.csv
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
