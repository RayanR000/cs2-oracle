# CS2 Oracle — Backend

Run everything from this directory through `venv/bin/python`. Note that `.env` here
points at production Supabase — see the root `AGENTS.md` gotcha before running scripts.

## Gotchas

- **Training data comes from Parquet, not the DB.** `fetch_price_history(backfilled_only=True)`
  reads `price-archive/*.parquet` via DuckDB. The DB supplies only the `is_backfilled`
  flag and events metadata.
- **Operational tables live in `price-archive/ops/*.parquet`.** API routes read Parquet
  first with a DB fallback. See `db/parquet.py`.
- **Nested values are stored as JSON text, store-wide.** `db/parquet.py::_jsonify_nested`
  serialises dict/list columns before they reach DuckDB, because DuckDB infers a nested
  column's SQL type *from the batch's contents* — the same code produced `STRUCT`,
  `MAP(VARCHAR, DOUBLE)` and `VARCHAR` for `prediction_accuracy.metrics` on different
  days, and the frozen on-disk struct rejected all but one. Never hand raw dicts to
  `append_table`. See `tests/test_parquet_nested_columns.py`.
- **`fetch_price_history` caches the voted frame** to `data/voted_<key>.parquet`
  (gitignored), keyed on the `prices-*.parquet` fingerprint + cutoff date + backfill slug
  set. **Bump `ItemForecaster.VOTED_CACHE_VERSION` if you change `_fetch_voted_price_history`
  or `_apply_multi_source_voting`** — the key cannot see code changes. `VOTED_CACHE=0` disables.
- **Training subsamples the pool.** `TRAIN_FEATURE_ROWS` (default
  `DEFAULT_TRAIN_FEATURE_ROWS = 100_000` in `scripts/forecast_prices.py`) caps the feature
  rows, so a retrain sees a small fraction of the item universe. Raising it is a measured
  multi-fold cost increase, and the fresh-model gate cannot detect the difference.
- **`MIN_SERVED_PRICE_USD = 1.0` is a convention, not a derivation.** `api/serving_policy.py`
  sets the floor deliberately equal to the lower bound of `HEADLINE_MIN_TIER` so the
  population the product shows is the population the headline accuracy number describes.
  `tests/test_serving_policy.py` fails if the two diverge. Sub-$1 items are ~72% of the
  forecast universe and one cent there is a 20% move.
- **Training is fully sequential.** Horizons, quantiles, and ensemble members train one at
  a time; LightGBM's OpenMP threads supply the CPU parallelism. Ensemble members get
  `n_jobs = max(1, cpu_count // 2)`; the Optuna search params still use `n_jobs: -1`.
- **The backtest resolves BOTH legs through `backtest/price_resolution.py::resolve_anchors`.**
  `item_forecasts.current_price` is stored on the outcome for reference and is **never
  scored on** — using it as the base leg is the bug that let one cohort score 61.76% and
  33.74% on different days. Resolved outcomes are **frozen**: `base_price` / `actual_price`
  / `resolved_at` are final, and `--reresolve` is the only thing that can move them
  (`--rescore` recomputes verdicts from the frozen actuals without reading the archive).
- **Backtest maturity is bounded by archive coverage, not `date.today()`.** A forecast is
  evaluable only once `prices-*.parquet` covers its target date; the cutoff is
  `min(today, archive_max_day())`. The archive always lags the calendar, and admitting
  that lag window puts guaranteed misses into the cohort — it tripped the 10%
  unresolvable gate at 38.5% and reported nothing.
- **`scripts/walkforward_backtest.py` does NOT use `fetch_price_history`.** Its own
  `_load_all_prices` skips multi-source voting, the `historical_fallback:` source filter,
  the dead-item filter, and `backfilled_only`. Lags are *not* corrupted — `engineer_features`
  collapses to one row per item-day — but it collapses the archive's 1.37× duplicate
  item-days with a plain **mean**, where production serves an **outlier-voted median**
  (sources >2σ from the median are rejected). The fresh-model gate therefore scores a
  different price consensus over a different item universe than production trains on.
  Not directly comparable to production DA.
- **Social sentiment features are permanently zero in production.** `reddit-sentiment.yml`
  was deleted: old.reddit.com returns `403 Blocked` from runner IPs, and prod
  `social_mentions` holds 0 rows all-time. The features also rank outside the top 20 of
  122 at every horizon (`docs/changelog/2026-07-22-social-feature-audit.md`), so don't
  rebuild the collector. `collectors/social_sentiment.py` is kept for local/authenticated
  runs and scores with **FinBERT ONNX INT8** — the "VADER" comments in `models/forecaster.py`
  and `database.py` are stale.
- **Model size/speed levers are documented.** See `docs/architecture/model-optimization.md`
  for the options that retain ≥90% quality.
