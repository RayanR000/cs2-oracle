# Data Collection Pipelines

One collector feeds the archive. Everything else is chained off it or runs on its own
cron. Five workflows exist in `.github/workflows/`:

| Workflow | Trigger | What it does | Writes |
|----------|---------|--------------|--------|
| `aggregator-update.yml` | cron `0 23 * * *` | Fetch 7 CSGOTrader price endpoints + exchange rates | `prices-YYYY-MM.parquet`, `exchange-rates-YYYY.parquet`, `collection_runs` |
| `price-forecast.yml` | `workflow_run` on *Aggregator Market Update* | Predict, and retrain when the model is ≥14 days old | `ops/item_forecasts.parquet` |
| `backtest-accuracy.yml` | `workflow_run` on *Price Forecast*, plus cron `0 8 * * 1-6` | Resolve matured forecasts, score them | `ops/forecast_outcomes.parquet`, `ops/prediction_accuracy.parquet`, `ops/accuracy_alerts.parquet` |
| `event-correlation-analysis.yml` | cron `0 4 * * 0` | Rebuild event/item impact correlations | `ops/events.parquet`, `ops/event_impacts_denorm.parquet` |
| `discover-new-items.yml` | `workflow_dispatch` only | Steam item discovery — **broken at import**, see below | nothing |

All four archive-writing workflows check out `RayanR000/cs2-oracle-data` and force-push it
back as a squashed orphan commit. **`CS2_DATA_REPO_TOKEN` is a required secret** for all
four (`aggregator-update.yml:91`, `price-forecast.yml:63`, `backtest-accuracy.yml:56`,
`event-correlation-analysis.yml:54`); without it the checkout fails before any work runs.

---

## CSGoTrader Multi-Market Aggregator

### Data Sources (from `prices.csgotrader.app/latest/`)

`csgotrader_aggregator.py:141-149` fetches 7 price endpoints; `fetch_exchange_rates()`
(:370) fetches an 8th, `exchange_rates.json`. The 7 endpoints fan out to **11 distinct
source labels** per day (`pipeline.py:131-143`), because `steam.json` carries four price
windows and `buff163.json` carries both a sell and a buy side.

| Endpoint | Source label(s) written | Field(s) used |
|----------|-------------------------|---------------|
| steam.json | `aggregator_sync`, `aggregator_steam_7d`, `aggregator_steam_30d`, `aggregator_steam_90d` | `last_24h` (with 7d/30d/90d fallback), `last_7d`, `last_30d`, `last_90d` |
| skinport.json | `aggregator_skinport` | `starting_at` |
| buff163.json | `aggregator_buff163`, `aggregator_buff163_buy` | `starting_at.price`, `highest_order.price` |
| csfloat.json | `aggregator_csfloat` | `price` |
| csmoney.json | `aggregator_csmoney` | `price` |
| csgotrader.json | `aggregator_csgotrader` | `price` |
| youpin.json | `aggregator_youpin` | `price` |
| exchange_rates.json | — (own Parquet file) | currency → rate |

The `aggregator_sync` fallback chain is why that label never goes missing when Steam has
no 24-hour print. Skinport reads `starting_at`, not `last_24h` — the earlier field choice
was a bug.

**Seven of the 11 labels vote in the consensus price.** `aggregator_buff163_buy` is a bid
(`highest_order`) and `aggregator_steam_7d/30d/90d` are Steam trailing-window MEAN sale
prices — the wrong side of the book and the wrong time basis, respectively — so all four are
collected but **excluded from consensus voting**, the bid as of 2026-08-07
(`models/item_parser.py::BID_SOURCES`) and the trailing-window means as of 2026-08-09
(`models/item_parser.py::TRAILING_WINDOW_SOURCES`). All four are still written to the archive
under their own label, so they are recoverable at read time. See
`docs/changelog/2026-08-07-bid-source-excluded-from-voting.md` and
`docs/changelog/2026-08-09-trailing-window-sources-excluded.md`.

### Files
- **`collectors/csgotrader_aggregator.py`** — one session for all endpoints; returns
  `{source: {item_name: raw_dict}}`. Logs per-endpoint failures and escalates to
  `CRITICAL` when all 7 fail.
- **`collectors/pipeline.py`** — maps sources to labels (:131-143), writes every
  item-source pair to the snapshot CSV (:273-352). Historical fallback rows are relabelled
  `historical_fallback:<source>` (:39-43) and stale items are tracked (:183-238), so
  carried-forward prices are distinguishable downstream. Validation is a bare `price > 0`
  check (:160) — `collectors/data_validation.py` is imported by nothing but its own test.
- **`collectors/snapshot_date.py`** — resolves the snapshot day **once**, in the workflow
  (`aggregator-update.yml:63-68`), and pins it into `AGGREGATOR_SNAPSHOT_DATE` for every
  later step. Before this, a run straddling midnight read the clock twice and silently
  dropped whole days from the archive.
- **`scripts/append_to_parquet.py`** — collapses the snapshot CSV to daily OHLCV and
  appends. Partitions prices and snapshots **by month** (:5-9) to stay under GitHub's
  100 MB per-file limit; exchange rates stay yearly.
- **`scripts/run_task.py`** — exits 1 when `items_collected == 0`, which files a labelled
  failure issue.

### Storage Strategy
- **Parquet is the record.** `price-archive/` (a gitignored local symlink to a checkout of
  `cs2-oracle-data`) holds all price data plus an `ops/` layer of operational tables.
- **Supabase** gets only a `CollectionRun` row per run. Prices in `price_history` are
  stale and not written by this pipeline.

See `docs/architecture/data.md` for the full archive layout.

### Coverage Per Run
- **~362,586 OHLCV rows/day** measured 2026-08-01, spread across the 11 source labels at
  roughly 29–34K rows each. `prices-2026-07.parquet` alone is 56 MB.

---

## Price Forecast

Chains off the aggregator (`workflow_run`, `price-forecast.yml:6-11`) and skips itself if
the upstream run did not succeed (:32). `timeout-minutes: 180` is the only hang protection
in the system — there are no code-level timeouts.

Mode is decided by "Determine run mode" (:70-86): Monday sets `mode=full`, but `full`
**only retrains if the model is ≥14 days old** or `FORCE_RETRAIN=1`
(`forecast_prices.py:210,227`). Drift is report-only unless `ALLOW_DRIFT_RETRAIN=1`
(:243-261). Monday is not a guaranteed retrain.

Two load-bearing steps beyond the obvious: "Publish updated archive" (:158-167) runs
*before* "Verify forecasts were persisted" (:174-180,
`scripts/check_forecast_freshness.py`), so the freshness check reads what actually landed
in the archive rather than passing on an unpublished local write.

`SKIP_CV=1` is deliberately absent from CI (:119-124). It is a local/dispatch speedup only.

---

## Backtest Accuracy

Chains off the forecast run and also carries its own cron (`0 8 * * 1-6`) so a failed
forecast day does not skip scoring. Resolves both the base and actual legs through
`backtest/price_resolution.py::resolve_anchors`; maturity is bounded by archive coverage
(`min(today, archive_max_day())`), not by the calendar. An unresolvable rate above
`MAX_UNRESOLVABLE_PCT = 10.0` (`backtest/resolution_gate.py:87`) reports nothing rather
than a biased number.

Failures auto-file a labelled GitHub issue (:109-123).

---

## Event Correlation Analysis

Weekly, Sundays 04:00 UTC. Rebuilds `events` and the denormalised `event_impacts_denorm`
Parquet the API reads. Independent of the daily chain.

---

## Discover New Items

Dispatch-only; the schedule was removed 2026-07-08. **It cannot run**:
`scripts/discover_steam_items.py:20` imports `from collectors.real_data_collector import
get_collector` and no `real_data_collector.py` exists, so a dispatch dies with
`ImportError` before any Steam request. There is currently no working path to add items to
the catalog — the CSMarketAPI backfill that was supposed to be the alternative has a
permanently exhausted free-key quota.

---

## Removed pipelines

- **Supply scraper** (`supply-scraper.yml`, deleted in `0288568`) — hosted GitHub runners
  are 429'd by Steam on the first request. `supply_snapshots` is frozen at 35,037 rows;
  its features were already excluded from training. See
  `docs/changelog/2026-07-16-drop-supply-depth.md`.
- **Reddit social sentiment** (`reddit-sentiment.yml`, deleted) — old.reddit.com returns
  `403 Blocked` from runner IPs, and the features were refuted independently. Do not
  rebuild it. `collectors/social_sentiment.py` survives for local/authenticated runs and
  scores with FinBERT ONNX INT8, not VADER. See
  `docs/changelog/2026-07-22-social-feature-audit.md`.
- **Player-count collector** — removed in `181488b`. The `player-counts-YYYY.parquet`
  files in the archive are a frozen historical dataset with nothing appending to them.

One-time merges (HuggingFace CS2 hourly dataset, CSMarketAPI backfill) are recorded in
`docs/changelog/2026-07-20-hf-dataset-merge.md` and are not part of any recurring run.

---

## Test Coverage
- **714 tests** across 41 files in `backend/tests/`.
- Run **`pytest tests`**, not bare `pytest` — `scripts/test_social_signal.py:25` imports
  `thefuzz`, which is not in `requirements.txt`, and collection aborts on it.
