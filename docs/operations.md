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
| `aggregator-update` | 23:00 UTC daily | Full item data collection from CSGOTrader (7 sources) | Parquet (`data-archive` branch), `collection_runs` |
| `price-forecast` | Chained off aggregator | ML price predictions (full retrain Mondays) | `item_forecasts` |
| `backtest-accuracy` | Chained off forecast + 08:00 UTC Mon-Sat | Evaluate forecast accuracy, detect concept drift | `prediction_accuracy`, `forecast_outcomes`, `accuracy_alerts` |
| `event-correlation-analysis` | Weekly Sun 04:00 UTC | Quantifies market-event price impacts | `event_correlations`, `event_impacts` |
| `discover-new-items` | Manual dispatch only | Steam discovery — dead (catalog curated via backfill) | `items` |

**Deleted 2026-07-31, do not resurrect from CI.** `supply-scraper` (Steam 429s runner
IPs on the first request) and `reddit-sentiment` (`old.reddit.com` returns 403 to
datacenter IPs) both reported success daily while storing zero rows. Neither is
repairable from hosted runners, and both served features already dropped. See
`docs/changelog/2026-07-31-accuracy-work-closed.md`. `supply_scraper.py` /
`social_sentiment.py` are kept for local or authenticated runs.

### Data flow

```
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

Green means the process exited 0, not that rows were written — always pair this with the
freshness query below.

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

### Verify output-table freshness (do this instead of trusting a green run)

```sql
SELECT 'item_forecasts'      AS t, max(forecast_date)   AS latest, count(*) AS rows FROM item_forecasts
UNION ALL SELECT 'prediction_accuracy', max(evaluation_date), count(*) FROM prediction_accuracy
UNION ALL SELECT 'event_correlations',  max(created_at)::date, count(*) FROM event_correlations;
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
- Forecast chains off aggregator automatically, ~17 min (predict-only, includes the
  chunked predict passes) or ~53 min (Monday retrain with regime models); a healthy
  run logs `Wrote 34,764 forecasts` for 8,691 items
- Backtest chains off forecast automatically, ~1-2 min
- A/B regime comparison: `python scripts/forecast_prices.py --compare-regime` (writes `lgbm-v3-regime` + `lgbm-v3-global-only` forecasts, runs backtest)
- The price-forecast workflow has `timeout-minutes: 180` as the last-resort safety net — if the code-level hang protections (30 min ensemble timeout, 2h horizon pool timeout, 10 min Optuna/CV timeout) are all bypassed, GHA kills the run after 3 hours
- All tables (`item_forecasts`, `prediction_accuracy`, etc.) stay bounded by UPSERT
- Parquet archive on `data-archive` grows by ~300 KB/day

### Warning signs

- ❌ Frequent failures — check the auto-created issues
- ⏳ Runs missing at expected times — GitHub Actions may be degraded
- Aggregator collecting 0 items — likely CSGOTrader upstream issue
- Accuracy tables not growing — forecast job may have failed; check logs
- **A green badge is not evidence of collection.** Three workflows reported success for
  weeks while storing nothing (see the deletion note above). Audit by querying output-table
  freshness, not by reading Actions. `run_task.py` now fails on zero rows for every count
  field a task returns, but only tasks that go through it are covered.

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
   - **Out of memory** — the predict phase is chunked (`PREDICT_CHUNK_ITEMS`, default
     1000, `0` disables); lower it before touching the training window
   - **`No boosters restored from the Actions cache`** — see below

### Forecast models are missing

Boosters are **not** in git — `*.txt` / `*.pkl` are gitignored, and committing them is
what produced the 1.8G unpushable-`main` blob purged in `47d3195`. They ride the Actions
cache:

- saved after `full` and `train-only` runs under key `forecast-models-<run_id>`, with the
  save step deliberately **not** conditioned on the forecast step succeeding (training's
  boosters must survive a later-phase crash)
- restored before `predict-only` via the `forecast-models-` prefix restore-key, so daily
  runs pick up the newest entry; the daily restore also keeps it warm against the 7-day
  eviction window
- `engineered_data.parquet` is excluded — it is a ~2 GB predict-side cache, which keeps
  the entry near 16 MB

If `Verify models are present` fails, re-run via `workflow_dispatch` with `mode=full` to
retrain and repopulate. `mode=train-only` repopulates without touching predict.

### Data not saving

- Verify `SUPABASE_DATABASE_URL` is correct
- Check `alembic current` matches the latest migration
- Run `python scripts/run_task.py migrate` manually

### Workflows without concurrency / failure notification

| Workflow | Missing concurrency | Missing failure notification |
|----------|:-------------------:|:---------------------------:|
| `backtest-accuracy` | ✅ absent | — |

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

# Event correlations (weekly job, run by hand)
python scripts/run_task.py event_correlation
```

### Backtest: frozen outcomes and the two escape hatches

Resolved outcomes are frozen — `base_price`, `actual_price` and `resolved_at` are
final, and the daily run cannot move them. That is what stops a green run from
silently restating history.

| Flag | Reads archive? | Moves frozen actuals? | Use when |
|---|---|---|---|
| *(none)* | only for forecasts with no outcome yet | no | the daily run |
| `--rescore` | no | no | a *scoring* change (verdict columns) must be applied to existing outcomes |
| `--reresolve` | yes, whole cohort | **yes** | the estimator or the archive changed and history must be rebuilt |

`--reresolve` is also the only path that deletes: outcomes whose forecast no
longer resolves are removed from the DB *and* the Parquet mirror.

**The unresolvable-rate gate.** If more than `MAX_UNRESOLVABLE_PCT` (10%) of the
cohort cannot be resolved, the run raises instead of reporting a number over a
silently shrunken cohort. When it fires, the cause is almost always archive data,
not code:

- **Missing days inside the window.** A target date with no observation makes the
  actual leg's window fall back onto pre-forecast days, and the disjoint-window
  rule rejects the pair. One missing day (`2026-07-22`) rejected an entire
  5,542-forecast cohort. Check `select day, count(*) from prices-*.parquet group
  by day` around the forecast dates in play.
- **Archive lag.** Maturity is capped at the archive's last day, so lag alone
  should not trip the gate — the run logs `Archive covers through <date> ... (Nd
  of lag)` when it clamps. If that line shows a large lag, the aggregator is
  behind.

**Rehearsing against SQLite.** `alembic upgrade` cannot replay from scratch on
SQLite: revisions `0001`→`0018` contain Postgres-only `ALTER COLUMN ... TYPE`
DDL. Use `alembic stamp 0018` on the snapshot first, then upgrade. Note that a
rehearsal still rewrites `price-archive/ops/*.parquet`, which is shared — back
those up first.

The collectors for the two deleted workflows still run locally, where residential IPs are
not blocked: `python scripts/run_supply_scraper.py` and
`python scripts/run_task.py reddit_social`. Neither feeds a live feature.

Note: a bare `pytest` from `backend/` currently aborts during collection —
`scripts/test_social_signal.py` is a one-off analysis script (not a test) that imports
`thefuzz`, which is not in `requirements.txt`. Run `pytest tests` (313 pass) until it is
renamed or the import is guarded.
