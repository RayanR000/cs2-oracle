# 2026-10-10 — API read paths: indexes, gzip, a freshness floor on opportunities

Performance review 2026-10-08, API section. No change to forecasts, bands or scoring.

## Indexes (migration 0030)

- **Dropped `idx_forecast_item_date`** (item_id, forecast_date, horizon_days). It has the
  same columns as `uq_item_forecast_date_horizon`, which Postgres already backs with an index,
  so every forecast write maintained two copies of it.
- **Added `idx_forecast_horizon_date`** (horizon_days, forecast_date DESC) INCLUDE (item_id).
  The API's "latest forecast per item at horizon h" reads filter on horizon and a
  forecast_date floor. Nothing led on horizon_days before. The model declares the plain
  column list; the DESC/INCLUDE spelling is Postgres-only and lives in the migration
  (schema drift reports index differences as warnings, not failures).
- **Added `idx_outcome_evaluated_at`** on `forecast_outcomes`. `/accuracy/outcomes` and the
  DuckDB fallback order by `evaluated_at DESC LIMIT n`, and both existing outcome indexes
  lead on another column.

The Aggregator's `migrate` step applies 0030 on its next run. The builds are plain
`CREATE INDEX`, which blocks writes to the table but not reads while it runs. The
Aggregator runs before the forecast job writes, so nothing is waiting on them.

## Opportunities

- **`_latest_forecasts` has a freshness floor**, the same `MAX_ARCHIVE_LAG_DAYS` (7) window
  `/items/trending` uses. Before, the distinct-on walked every forecast ever written at the
  horizon, and an item that left the forecast universe kept ranking on its last band, however
  old. The constant moved from `api/routes/items.py` to `api/serving_policy.py` so both
  routes share it.
- **`/opportunities/momentum` is cached** (300 s, like `/opportunities/`), and **rejects a
  horizon outside `SERVED_HORIZONS` with 400**, as `/items/volatility` already does. It used
  to accept any 1–30 and return an empty list for the 26 horizons with no rows.

## Responses

- **`GZipMiddleware` at level 5**, for responses of 1 KB or more. `/accuracy/summary` is
  ~1.7 MB raw and ~120 KB gzipped. Level 5 measured 6 ms against 29 ms at level 9.
- **`/accuracy/` gets the same Cache-Control as the other public data routes**
  (`max-age=300, stale-while-revalidate=600`). It was missing from the prefix list.

## Not done

From the same section: the accuracy routes still fetch 500–2,000 rows to keep ~4; there is
still no ETag/304; and `get_or_build` still has no single-flight lock and a fixed TTL
rather than a data-version key.

Tests: `tests/test_api_read_paths.py`.
