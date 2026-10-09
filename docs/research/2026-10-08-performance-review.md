# Performance review — 2026-10-08

Read-only review of the whole pipeline for speed and efficiency, plus a scan of open-source
projects worth borrowing from. Six parallel reviewers: `forecaster.py`; collectors and the
Aggregator; forecast/backtest scripts; API; tests and CI; open-source.

**Evidence levels.** *Measured* = timed locally on an M4 against the local archive or the
synthetic fixture. *Estimated* = inferred from code, row counts or CI log timestamps. Nothing
touched Supabase or GitHub, so every prod query cost is an estimate. ✔ = the claim was
re-checked against the code after the review.

## Output-identical wins (no retrain)

### 1. Backtest: 28–35 min/day → est. under 10 min

The longest step in the daily chain. Cost is O(all history) per day for ~22K new rows.

- ✔ `scripts/backtest_accuracy.py:1116` pulls every `item_forecasts` row and filters maturity in
  Python.
- ✔ `backtest/candidate_resolution.py:131-142` loads `ForecastCandidate`,
  `ForecastCandidateOutcome` and `ForecastOutcome` whole, as ORM objects. Candidates grow
  ~44K/day — an OOM trend as well as a speed one.
- `forecast_outcomes` (~800K rows) is read 4–5 times in 900-row chunks (~3,500 statements):
  `:1149`, the `:532` existence recheck, `_refresh_verdict_columns` (`:1448`),
  `_records_from_frozen_outcomes` (`:1456`).
- ✔ `backtest/price_resolution.py:243` votes with the pandas `_apply_multi_source_voting`, not
  the DuckDB `_multi_source_voting_sql` the forecaster moved to on 2026-09-28 (177 s → 5–15 s).
  ✔ It is called once per `(horizon, model_version)` group (`backtest_accuracy.py:1230`). Each
  of the 4 passes takes ~5 min in run 37719560908.
- ✔ `update_bias = True` by default (`:1519`). The job never restores `saved_models`, and the
  `bias_corrections.json` it writes is discarded with the runner.

Fix: anti-join to fetch only unfrozen mature forecasts and unresolved candidates; one streamed
refresh-and-score read; vote once via DuckDB for the union window, then slice back per group
before `stale_run_lookup`; default `update_bias` off. The SQL vote differs from pandas on 17 of
6.58M rows, all sub-$1 — needs an equivalence test and a changelog note.

Smaller items in the same job:
- `penny_metrics` (`:987`) recomputes the tier-0 cohort already in `tiered`.
- Chunking `bootstrap_ci` draws: measured −4 GB peak, identical CI bounds.
- `_upsert_accuracy` and `candidate_store.write_candidate_batches` write row by row.
- The evidence-report steps run serially and drop 72% of rows (sub-$1) in Python, not in SQL.

### 2. Predict feature engineering: est. ~2 min → ~12 s

- ✔ `_attach_sidecars` (`models/forecaster.py:5085`) merges supply-history (15.4M rows),
  stattrak-panel (4.5M) and bid-panel on every call; only the volume panel is gated. No served
  feature reads them (`BID_FEATURES`, `STATTRAK_FEATURE`, `SUPPLY_CHURN_FEATURES` all off).
  Measured: 79% of predict feature time.
- `_engineer_features_chunked` (`:9555-9613`) omits `skip_unused_groups`, and runs a pass A
  only to build `cross_sectional` / `supply_depth` columns that serving discards. Safe to skip
  only while `regime_models` is empty and the tier-lead / xs-rank flags are off.
- Measured on 2,000 items: 44.6 s → 2.9–4.3 s, served columns bit-identical.
- Training side: `engineer_features` 7.8 s → 3.1 s on 640k rows.

Also, smaller:
- `_fetch_prior_forecasts` (`:9413`) pulls all history per horizon to keep the latest row per
  item; use `DISTINCT ON`.
- `_sanitize_forecasts` (`:10326`): measured 1.15 s → 0.08 s.
- The ranking head evaluates NDCG on a validation set it never reads (`:8164`).
- `prepare_targets` recomputes horizon-independent work per horizon (~5 s per retrain).

### 3. CI test gate: ~80 s → ~35 s

- `OMP_NUM_THREADS=1` plus pytest-xdist `-n 4`. Measured locally 79 s → 32 s; all 2,769 tests
  pass at `-n 2`, `-n 4` and `-n auto`. pytest-xdist is not in the `dev` extra yet.
- Local runs take 182 s because `engineer_features` reads the real `price-archive` sidecars
  (~6.5 s per call; `archive_dir` defaults to the real directory at `forecaster.py:1101`). An
  autouse fixture pointing it at a temp dir brings local to ~79 s and makes the tests hermetic.
- `test_tied_anchor_cohort.py::_run_cv` repeats an identical CV fit ~8× (18.6 s).
- `lint.yml` `paths:` omits `.github/workflows/**`, though ≥6 test files read workflow YAML.

### 4. Archive writes

- `scripts/append_to_parquet.py:243-281` rewrites the month through pandas. Measured 6.4 s /
  5.7 GB → 0.7 s / 0.46 GB with a DuckDB anti-join, identical row set. Use `least()` on
  `ingested_at` to keep first-arrival. Peak memory reaches ~7 GB at month end.
- ZSTD plus `ORDER BY source, item_slug, day`: measured 168 MB → 104 MB (−38%) across the
  price files. ZSTD alone, with no reordering risk, is −26%.
- `collectors/pipeline.py:181-189`, `:243-250` build ~380k `PriceHistory` ORM objects that are
  never persisted. Measured 2.1 s and +457 MB.

### 5. Dead or redundant workflow work

- ✔ **Done 10-09.** "Run drift detection" (`price-forecast.yml:293`) needs `evidently`, which only the `mlops`
  extra installs; it ImportErrors daily behind `continue-on-error`. Its output has no consumer.
- **Done 10-09 (removed).** The voted-frame Actions cache (`price-forecast.yml:124`) can never hit: the in-code key holds
  the cutoff date and the archive row counts, which change daily.
- **Done 10-09.** The backtest weekday cron re-freezes what the chained run already did (~25 runner-min/day).
  Replace it with an early exit, since it is also the fallback when Price Forecast fails.
- **Timeout done 10-09.** The Aggregator could run supply and volume collection in parallel with aggregate/append
  (est. 1–1.5 min), and needs `timeout-minutes` on the supply step.
- **Unused imports dropped 10-09.** The Aggregator installs ML/serving dependencies it never imports (scipy, onnxruntime,
  transformers, lightgbm, optuna, fastapi). `transformers`, `onnxruntime`, `huggingface_hub`,
  `joblib`, `beautifulsoup4` and `apscheduler` are imported nowhere.

## Correctness findings

- **No retry or backoff in the daily ingest** (`csgotrader_aggregator.py:41`, `:161-165`;
  `supply_depth.py`; `sales_volume.py`; the first DB read in `pipeline.py:113`). The 2026-08-23
  run died on one `OperationalError`. A single failed feed is skipped and the run still goes
  green, and the dumps are live-only, so that source-day cannot be recovered.
- ✔ **`/items/trending` ordering** (`api/routes/items.py:158-165`): an outer join plus
  `ORDER BY ratio DESC` puts NULLs first on Postgres, so items without a forecast probably lead
  the list. SQLite sorts NULLs last, so the tests miss it. Fix with `.nulls_last()` or an inner
  join. The same route also ignores `subq.c.current_price` and runs `_latest_prices` anyway.
- ✔ **The 30d calibration is fitted on a different model than the one served.** CV forces
  `objective="quantile"` at every horizon (`forecaster.py:10558`), while `CENTRE_OBJECTIVE`
  trains the served 30d centre as regression. The 30d OOF residuals behind `q_hat` and the
  calibrators therefore come from an MAE model. **Needs a retrain plus a coverage check.**
- **Done 10-09 (installed by hand; migration 0029).** `pg_trgm` is created by no migration, yet `api/routes/market.py:70` calls `similarity()`.
  Either it was installed by hand (schema drift) or the `q` path errors in prod.
- `_fetch_prior_forecasts` has no recency cap; a prior of any age is blended at weight 0.15.
- `_engineer_features_chunked` never writes the engineered cache, so with more than 1,000
  eligible items the predict-path cache is dead code.

## API

Every route is a sync `def`, so nothing blocks the event loop, and there is no N+1.
`price_history` is stale (the aggregator writes Parquet only), so those routes' costs are latent.

- ✔ `idx_forecast_item_date` duplicates `uq_item_forecast_date_horizon` (`database.py:222-223`).
  Drop it, and add `(horizon_days, forecast_date DESC) INCLUDE (item_id)`.
  `_latest_forecasts` in `opportunities.py:68-96` also has no date floor.
- `/opportunities/momentum` is uncached, and accepts horizons with no rows.
- `get_or_build` has no single-flight lock, and the TTL is 300–600 s for data that changes daily.
  Key the cache on a data version instead.
- The accuracy routes fetch 500–2,000 rows to keep ~4. `/summary` is 1.7 MB raw (~120 KB
  gzipped), and `/accuracy/` is missing from the Cache-Control prefixes.
- There is no `GZipMiddleware` and no ETag/304. Use level 5: measured 6 ms vs 29 ms at level 9.
- `forecast_outcomes` has no index leading on `evaluated_at`.

## Open-source projects

Nothing external would move forecast accuracy, and the best candidates are already adopted:
statsforecast, MAPIE, scoringrules, DuckDB, Conformal PID.

1. **`bashtage/arch`** (block bootstrap), or a Diebold–Mariano test with the HLN correction on
   the in-repo `hac_long_run_variance`. Do this before the 10-23 / 10-25 reads.
   `backtest/candidate_scoring.py::paired_daily_interval` and `scoring.py::block_bootstrap_ci`
   resample consecutive forecast dates as iid, but adjacent dates share h−1 forward days. A
   worst-case simulation (not the real panel) put nominal 90% coverage at 36% for h=7. Measure
   the real lag-1..h autocorrelation first. Add a sensitivity row rather than editing frozen
   preregs; `promotion.py` is not frozen.
2. **`salesforce/online_conformal`** (Apache-2.0, archived): SAOCP / SF-OGD / FACI. Port, do not
   depend, and only if the 10-23 PID read reopens adaptive conformal.
3. **Skip:**
   - Chronos and TimesFM: expected to lose to the naive band, and they need torch.
   - mlforecast: row-based lags break on gaps.
   - crepes, puncc, quantile-forest, Venn–Abers, Polars, orjson.
4. No free data source covers the 2026-04-16..07-08 gap.
5. `docs/references/open-source-shortlist.md` has two errors: the crepes link points to a dead
   stub (the live repo is `henrikbostrom/crepes`), and puncc is MIT.

## Suggested order

1. Backtest table reads and the DuckDB vote (§1).
2. Sidecar gating and the chunked predict pass (§2).
3. Ingest retries (correctness).
4. `/items/trending` NULL ordering.
5. CI test speed (§3).
6. The 30d CV objective. This one needs a retrain and backtest.
