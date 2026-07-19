# Workflow Monitoring Guide

## Overview

The backend runs entirely on GitHub Actions scheduled workflows. Most workflows have:

- **`set -o pipefail`** in every run step (failure = red, not green)
- **`concurrency` groups** (no overlapping runs)
- **`timeout-minutes`** (no stuck runs)
- **`shell: bash`** (consistent pipefail behavior)
- **Failure notification** via `gh issue create` on schedule-triggered failures

## Workflows

| Workflow | Schedule | Purpose | Writes to |
|----------|----------|---------|-----------|
| `supply-scraper` | 22:00 UTC daily | Steam sell_listings supply snapshots | `supply_snapshots` |
| `aggregator-update` | 23:00 UTC daily | Full item data collection from CSGOTrader (7 sources) | Parquet (`data-archive` branch), `collection_runs` |
| `player-count-hourly` | Every 2h | Steam player-count tracking | Supabase `player_counts`, Parquet archive |
| `price-forecast` | Chained off aggregator | ML price predictions (full retrain Mondays) | `item_forecasts` |
| `backtest-accuracy` | Chained off forecast + 08:00 UTC Mon-Sat | Evaluate forecast accuracy, detect concept drift | `prediction_accuracy`, `forecast_outcomes`, `accuracy_alerts` |
| `event-correlation-analysis` | Weekly Sun 04:00 UTC | Quantifies market-event price impacts | `event_correlations`, `event_impacts` |
| `discover-new-items` | Manual dispatch only | Steam discovery — dead (catalog curated via backfill) | `items` |

### Data flow

```
Every 2h  Player Count Hourly → Steam active players → player_counts (Supabase + Parquet)

22:00  Supply Scraper → Steam burst scrape → supply_snapshots (sell_listings)

23:00  Aggregator → prices → CSV → Parquet (data-archive branch)
        └─▶ Price Forecast (chained) → Parquet (all-time) → item_forecasts
              └─▶ Backtest Accuracy (chained) → prediction_accuracy
                                                  forecast_outcomes
                                                  accuracy_alerts

Sun 04:00  Event Correlation Analysis → event_correlations, event_impacts
```

## How to check workflow status

### GitHub UI

1. Go to your repository on GitHub
2. Click the **Actions** tab
3. Check for:
   - ✅ Green = successful
   - ❌ Red = failed (an issue should be auto-created)
   - ⏳ In progress

### Verify data collection

Run this in Supabase SQL Editor:

```sql
SELECT
    started_at,
    finished_at,
    status,
    total_items,
    successful,
    failed,
    duration_seconds
FROM collection_runs
WHERE started_at > now() - interval '7 days'
ORDER BY started_at DESC
LIMIT 20;
```

### Check the Parquet archive

The `data-archive` branch should have a new commit from each aggregator run:

```bash
git fetch origin data-archive
git log origin/data-archive --oneline -5
```

## Expected patterns

### Healthy state

- Aggregator runs once daily at ~23:00 UTC, ~5,525 items, ~60s
- Forecast chains off aggregator automatically, ~2-5 min (predict-only) or ~53 min (Monday retrain with regime models)
- Backtest chains off forecast automatically, ~1-2 min
- A/B regime comparison: `python scripts/forecast_prices.py --compare-regime` (writes `lgbm-v3-regime` + `lgbm-v3-global-only` forecasts, runs backtest)
- All tables (`item_forecasts`, `prediction_accuracy`, etc.) stay bounded by UPSERT
- `chart_points` never pruned — bounded at ~4M rows (one close per item per day)
- Parquet archive on `data-archive` grows by ~300 KB/day

### Warning signs

- ❌ Frequent failures — check the auto-created issues
- ⏳ Runs missing at expected times — GitHub Actions may be degraded
- Aggregator collecting 0 items — likely CSGOTrader upstream issue
- Player count hourly failing silently — check logs (no failure notification on this workflow)
- Accuracy tables not growing — forecast job may have failed; check logs

## Troubleshooting

### Workflow didn't run

- Check GitHub Actions status page
- Verify `SUPABASE_DATABASE_URL` is set in repository secrets
- Forecast/backtest chain off the upstream workflow — if upstream failed, downstream won't run

### Workflow failed

1. Check the auto-created issue (title includes the workflow name and date)
2. Download the logs artifact from the run
3. Common issues:
   - **`alembic upgrade head` fails** — schema drift; run manually against Supabase
   - **CSGOTrader API down** — aggregator returns 0 prices; check upstream
   - **Disk space** — the Parquet steps can grow the checkout on the runner
   - **Out of memory** — full retrain on all 8,691 items; try reducing the training window

### Data not saving

- Verify `SUPABASE_DATABASE_URL` is correct
- Check `alembic current` matches the latest migration
- Run `python scripts/run_task.py migrate` manually

### Workflows without concurrency / failure notification

| Workflow | Missing concurrency | Missing failure notification |
|----------|:-------------------:|:---------------------------:|
| `backtest-accuracy` | ✅ absent | — |
| `player-count-hourly` | — | ✅ absent |

`discover-new-items` has a failure notification step with a `schedule` trigger condition, but the workflow has no schedule trigger — the step can never fire.

## Manual testing

```bash
cd backend
source venv/bin/activate

# Full aggregator collection
python scripts/run_task.py aggregate

# Forecast (with saved models)
python scripts/forecast_prices.py --predict-only

# Forecast (full retrain)
python scripts/forecast_prices.py

# Forecast (regime A/B comparison)
python scripts/forecast_prices.py --compare-regime

# Backtest forecast accuracy
python scripts/backtest_accuracy.py --type forecast

# Player count snapshot
python -m collectors.player_counts

# Supply scraper
python scripts/run_supply_scraper.py
```
