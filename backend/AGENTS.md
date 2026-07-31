# CS2 Oracle — Backend

## Gotchas

- **Training data from Parquet, not DB.** `fetch_price_history(backfilled_only=True)` reads `price-archive/*.parquet` via DuckDB. DB only used for `is_backfilled` flag + events metadata.
- **Operational tables migrated to `price-archive/ops/*.parquet`.** API routes read Parquet first with DB fallback. See `backend/db/parquet.py`.
- **Social sentiment features are non-functional.** VADER scores CS2 jargon as neutral — features don't rank in top 20 by gain. See `docs/architecture/model-optimization.md` Tier-1 #1 for removal.
- **Training is fully sequential.** No multiprocessing or threading in the training path. Horizons, quantiles, and ensemble members all train one at a time. LightGBM's internal OpenMP threads handle CPU parallelism. `n_jobs` is set to `max(1, cpu_count // 2)`.
- **Model optimization options documented.** See `docs/architecture/model-optimization.md` for all levers to reduce model size/speed while retaining ≥90% quality.
- **`fetch_price_history` caches the voted frame** to `backend/data/voted_<key>.parquet` (gitignored). Keyed on the `prices-*.parquet` fingerprint + cutoff date + backfill slug set. **Bump `ItemForecaster.VOTED_CACHE_VERSION` if you change `_fetch_voted_price_history` or `_apply_multi_source_voting`** — the key cannot see code changes. `VOTED_CACHE=0` disables.
- **`walkforward_backtest.py` does NOT use `fetch_price_history`.** Its own `_load_all_prices` skips multi-source voting, the `historical_fallback:` source filter, the dead-item filter, and `backfilled_only`. Lags are *not* corrupted — `engineer_features` collapses to one row per item-day — but it collapses the archive's 1.37× duplicate item-days with a plain **mean**, where production serves an **outlier-voted median** (sources >2σ from the median are rejected). So the fresh-model gate scores a different price consensus over a different item universe than production trains on. Not directly comparable to production DA.
