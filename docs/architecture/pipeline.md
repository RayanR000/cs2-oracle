# Data Collection Pipelines

One collector feeds the archive. Everything else is chained off it, runs on its own cron, or
is dispatched by hand. Nine workflows exist in `.github/workflows/` (2026-10-13):

| Workflow | Trigger | What it does | Writes |
|----------|---------|--------------|--------|
| `aggregator-update.yml` | cron `0 23 * * *` | Fetch 7 CSGOTrader price endpoints + exchange rates; apply migrations | `prices-YYYY-MM.parquet`, `exchange-rates-YYYY.parquet`, `collection_runs`, `ops/items.parquet` |
| `price-forecast.yml` | `workflow_run` on *Aggregator Market Update* | Predict, and retrain when the model is ≥14 days old | `ops/item_forecasts.parquet`, `ops/item_forecasts/YYYY-MM.parquet`, `ops/forecast_candidates/YYYY-MM.parquet` |
| `backtest-accuracy.yml` | `workflow_run` on *Price Forecast*, plus cron `0 8 * * 1-6` | Resolve matured forecasts, score them, check Postgres/partition parity, publish closed months as releases | `ops/forecast_outcomes{.parquet,/YYYY-MM.parquet}`, `ops/forecast_candidate_outcomes/YYYY-MM.parquet`, `ops/prediction_accuracy.parquet`, `ops/accuracy_alerts.parquet` |
| `forecast-freshness-check.yml` | cron `0 12 * * *` | Independent of the chain: turns "no forecast landed today" into a red run, even on a night the forecast job never ran | nothing |
| `schema-drift-check.yml` | `pull_request` / `push` to main (paths-filtered) | Pre-merge gate: applies the migrations to a throwaway Postgres and diffs against `Base.metadata`, so an ORM column with no migration fails before merge | nothing |
| `lint.yml` | `pull_request` / `push` to main | `ruff`, `mypy`, import smoke test, fast test gate | nothing |
| `model-diagnostics.yml` | `workflow_dispatch` only | Score the served classifier — one matrix job per horizon, `--train-only`, artifacts discarded. A dispatch with default inputs is the control arm | nothing |
| `ops-sync.yml` | `workflow_dispatch` only | Copy named forecast dates from Postgres into the monthly partitions (fill ids, repair a failed night) | `ops/<table>/YYYY-MM.parquet` |
| `archive-restore-day.yml` | `workflow_dispatch` only | One-off repair of a deleted archive day | `prices-YYYY-MM.parquet` |

Every archive-writing workflow checks out `RayanR000/cs2-oracle-data` and force-pushes it back
as a squashed orphan commit with `--force-with-lease`, so a concurrent writer fails rather than
being overwritten. **`CS2_DATA_REPO_TOKEN` is a required secret** for each of them; without it
the checkout fails before any work runs. `forecast-freshness-check.yml` and
`model-diagnostics.yml` check the archive out read-only with the same secret.

Retired 2026-10-13: `event-correlation-analysis.yml`, `discover-new-items.yml` (broken at
import), `ab-harness-batch.yml` (all paused since 2026-08-23 for Actions minutes) and the
one-off `ops-partition-backfill.yml` (applied 2026-10-12). Their scripts remain; git history
keeps the YAML.

---

## CSGoTrader Multi-Market Aggregator

### Data Sources (from `prices.csgotrader.app/latest/`)

`csgotrader_aggregator.py:30-36` declares 7 price endpoints; `fetch_exchange_rates()`
(:387) fetches an 8th, `exchange_rates.json`. The 7 endpoints fan out to **12 distinct
source labels** per day (`pipeline.py:151-161` plus the raw-source pass at :317-380), because
`steam.json` carries four price windows *and* a fallback-free spot, and `buff163.json` carries
both a sell and a buy side.

| Endpoint | Source label(s) written | Field(s) used |
|----------|-------------------------|---------------|
| steam.json | `aggregator_sync`, `aggregator_steam_spot`, `aggregator_steam_7d`, `aggregator_steam_30d`, `aggregator_steam_90d` | `last_24h` (with 7d/30d/90d fallback), `last_24h` (no fallback), `last_7d`, `last_30d`, `last_90d` |
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

**Seven of the 12 labels vote in the consensus price.** `aggregator_buff163_buy` is a bid
(`highest_order`), `aggregator_steam_7d/30d/90d` are Steam trailing-window MEAN sale
prices — the wrong side of the book and the wrong time basis, respectively — and
`aggregator_steam_spot` is Steam's fallback-free `last_24h`, which would cast a *second*
Steam ballot beside `aggregator_sync` and double-count the venue. All five are collected but
**excluded from consensus voting**: the bid as of 2026-08-07
(`models/item_parser.py::BID_SOURCES`), the trailing-window means as of 2026-08-09
(`::TRAILING_WINDOW_SOURCES`), and the spot as of 2026-08-17
(`::STEAM_SPOT_SOURCES`, added for the cross-venue basis feature's Steam leg;
`tests/test_steam_spot_source.py`). All five are still written to the archive under their own
label, so they are recoverable at read time. See
`docs/changelog/2026-08-07-bid-source-excluded-from-voting.md`,
`docs/changelog/2026-08-09-trailing-window-sources-excluded.md` and
`docs/changelog/2026-08-17-steam-spot-persisted-for-basis.md`.

### Files
- **`collectors/csgotrader_aggregator.py`** — one session for all endpoints; returns
  `{source: {item_name: raw_dict}}`. Logs per-endpoint failures and escalates to
  `CRITICAL` when all 7 fail.
- **`collectors/pipeline.py`** — maps sources to labels (:151-161), writes every
  item-source pair to the snapshot CSV (:291-302, plus the raw-source pass at :317-380).
  Historical fallback rows are relabelled `historical_fallback:<source>` (:55-59) and stale
  items are tracked (:206-262), so carried-forward prices are distinguishable downstream.
  Validation is a bare `price > 0` check (:179) — `collectors/data_validation.py` is imported by `collectors/__init__.py` (though nothing calls the validation functions in the live pipeline).
- **`collectors/snapshot_date.py`** — resolves the snapshot day **once**, in the workflow
  (`aggregator-update.yml:84-90`), and pins it into `AGGREGATOR_SNAPSHOT_DATE` for every
  later step. Before this, a run straddling midnight read the clock twice and silently
  dropped whole days from the archive.
- **`scripts/append_to_parquet.py`** — collapses the snapshot CSV to daily OHLCV and
  appends. Partitions prices and snapshots **by month** (:5-9) to stay under GitHub's
  100 MB per-file limit; exchange rates stay yearly.
- **`scripts/run_task.py`** — exits 1 when `items_collected == 0`, which files a labelled
  failure issue.

### Storage Strategy
- **Parquet is the record.** `price-archive/` (a gitignored **plain local directory** — not
  a symlink and not a checkout of `cs2-oracle-data`) holds all price data plus an `ops/`
  layer of operational tables. Editing it changes nothing in production: the durable archive
  is the separate `RayanR000/cs2-oracle-data` repo, which only CI writes.
- **Supabase** gets only a `CollectionRun` row per run. Prices in `price_history` are
  stale and not written by this pipeline.

See `docs/architecture/data.md` for the full archive layout.

### Coverage Per Run
- **~362,586 OHLCV rows/day** measured 2026-08-01, spread across the then-11 source labels at
  roughly 29–34K rows each; `aggregator_steam_spot` was added 2026-08-17 and is not in that
  figure. `prices-2026-07.parquet` alone is 56 MB.

---

## Price Forecast

Chains off the aggregator (`workflow_run`, `price-forecast.yml:7-10`) and skips itself if
the upstream run did not succeed (:34). `timeout-minutes: 180` is the only hang protection
in the system — there are no code-level timeouts.

Mode is decided by "Determine run mode" (:77-84): Monday sets `mode=full`, but `full`
**only retrains if the model is ≥14 days old** or `FORCE_RETRAIN=1`
(`forecast_prices.py:475-476,493`). Drift is report-only unless `ALLOW_DRIFT_RETRAIN=1`
(:513-529). Monday is not a guaranteed retrain.

Two load-bearing steps beyond the obvious: "Publish updated archive (flat history)" (:316)
runs *before* "Verify forecasts were persisted" (:332-338,
`scripts/check_forecast_freshness.py`), so the freshness check reads what actually landed
in the archive rather than passing on an unpublished local write.

`SKIP_CV=1` is deliberately absent from CI (:155). It is a local/dispatch speedup only.

---

## Backtest Accuracy

Chains off the forecast run and also carries its own cron (`0 8 * * 1-6`) so a failed
forecast day does not skip scoring. Resolves both the base and actual legs through
`backtest/price_resolution.py::resolve_anchors`; maturity is bounded by archive coverage
(`min(today, archive_max_day())`), not by the calendar. An unresolvable rate above
`MAX_UNRESOLVABLE_PCT = 10.0` (`backtest/resolution_gate.py:118`) reports nothing rather
than a biased number.

Failures auto-file a labelled GitHub issue (:129-137) — on `workflow_run` and `schedule` only; manual dispatches stay silent.

---

## Event Correlation Analysis

Weekly, Sundays 04:00 UTC. Rebuilds `events` and the denormalised `event_impacts_denorm`
Parquet the API reads. Independent of the daily chain.

---

## Discover New Items (retired 2026-10-13)

The workflow was deleted: its script imported a `collectors.real_data_collector` that no longer
existed, so every dispatch died with `ImportError`. Catalog growth now runs through
`collectors/new_item_discovery.py` inside the Aggregator, and the CSMarketAPI backfill (free-key
quota resets monthly).

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
- **2,427 tests** across 190 files in `backend/tests/` (collected 2026-09-15).
- A bare `pytest` collects cleanly now: the blocker (`scripts/test_social_signal.py`, which
  imported `thefuzz`) was deleted 2026-08-10. Still prefer a targeted
  `venv/bin/python -m pytest tests/test_<name>.py` — the full suite trains models.
