# Workflow Monitoring Guide

## Overview

The backend runs entirely on GitHub Actions scheduled workflows. Most workflows have:

- **`set -o pipefail`** in every run step (failure = red, not green)
- **`concurrency` groups** (no overlapping runs)
- **`shell: bash`** (consistent pipefail behavior)
- **Failure notification** via `gh issue create` on schedule-triggered failures

There is **no hang protection except one job-level timeout** — see
[Hang protection](#hang-protection-there-is-only-one).

## Required secrets

| Secret | Used by | If unset |
|---|---|---|
| `SUPABASE_DATABASE_URL` | all five workflows | every DB step fails |
| `CS2_DATA_REPO_TOKEN` | `aggregator-update`, `price-forecast`, `backtest-accuracy` | the **archive checkout fails before any collection or forecasting runs** — the job dies at `Checkout data archive` / `Checkout price archive` |

`CS2_DATA_REPO_TOKEN` must be able to read *and* force-push
`RayanR000/cs2-oracle-data` (`aggregator-update.yml:91`, `price-forecast.yml:63`,
`backtest-accuracy.yml:56`). `STEAM_API_KEY` and the `CSMARKETAPI_*` keys are optional;
nothing in the daily pipeline reads them.

## Workflows

| Workflow | Schedule | Purpose | Writes to |
|----------|----------|---------|-----------|
| `aggregator-update` | 23:00 UTC daily | Full item data collection from CSGOTrader (~11 source labels per day) | Parquet archive (`cs2-oracle-data` repo), `collection_runs` |
| `price-forecast` | Chained off aggregator | ML price predictions (Monday sets `mode=full`) | `item_forecasts` + its Parquet mirror |
| `backtest-accuracy` | Chained off forecast + 08:00 UTC Mon-Sat | Evaluate forecast accuracy, detect concept drift | `prediction_accuracy`, `forecast_outcomes`, `accuracy_alerts` |
| `event-correlation-analysis` | Weekly Sun 04:00 UTC | Quantifies market-event price impacts | `event_correlations`, `event_impacts` |
| `discover-new-items` | Manual dispatch only | **Broken at import — cannot run** | nothing |

`discover-new-items` is not merely dormant: `scripts/discover_steam_items.py:20` imports
`collectors.real_data_collector`, a module that does not exist. Any dispatch dies with
`ImportError` before a single Steam request. **Item onboarding has no working path today** —
Steam discovery is broken, and the CSMarketAPI free-key quota is permanently exhausted
(all keys still 429). Do not tell anyone the catalog can be extended.

**Deleted 2026-07-31, do not resurrect from CI.** `supply-scraper` (Steam 429s runner
IPs on the first request) and `reddit-sentiment` (`old.reddit.com` returns 403 to
datacenter IPs) both reported success daily while storing zero rows. Neither is
repairable from hosted runners, and both served features already dropped. See
`docs/changelog/2026-07-31-accuracy-work-closed.md`. `supply_scraper.py` /
`social_sentiment.py` are kept for local or authenticated runs.

### Data flow

```
23:00  Aggregator → prices → CSV → Parquet (cs2-oracle-data repo, force-pushed)
        └─▶ Price Forecast (chained) → item_forecasts + Parquet mirror
              └─▶ Backtest Accuracy (chained) → prediction_accuracy
                                                  forecast_outcomes
                                                  accuracy_alerts

Sun 04:00  Event Correlation Analysis → event_correlations, event_impacts
```

### Load-bearing steps that can each fail a run

These three are easy to miss in a log and each one is a hard failure:

1. **`Resolve snapshot date`** (`aggregator-update.yml:63-68`, `collectors/snapshot_date.py`)
   — resolves the archive day **once** and exports `SNAPSHOT_DATE` for every later step.
   Before it existed, a run started after midnight (the 23:00 cron has been observed
   starting at 00:08) stamped the pipeline and the append step from two different clock
   reads, silently losing `2026-07-27`, `2026-07-30` and `2026-08-03` from the archive.
   If a day is missing from the archive, check this step's logged value first.
2. **`Publish updated archive (flat history)`** — present in all three archive-writing
   workflows. Each job checks out `cs2-oracle-data` into an ephemeral `archive/`, so
   without this step the run's Parquet writes are discarded at teardown. This is exactly
   how `item_forecasts.parquet` froze at 2026-07-29 while Supabase kept advancing. It
   runs `if: always()` (except `train-only`) and force-pushes an orphan commit, so a red
   run can still have published. **In `price-forecast.yml` it is ordered before the
   freshness check on purpose** — reversed, the check would pass on state that does not
   survive the job.
3. **`Verify forecasts were persisted`** (`price-forecast.yml:174-180`,
   `scripts/check_forecast_freshness.py`) — fails the run unless *both* the DB and the
   Parquet mirror carry a forecast for the expected date. The API reads the mirror first
   and falls back to the DB, so a DB row without a mirror row serves nothing. A failure
   here means the forecast step "succeeded" while persisting nothing; do not re-run
   blindly, check which of the two stores is behind.

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

The archive is **not a branch of this repo.** It lives in a separate repo,
`RayanR000/cs2-oracle-data`, branch `main`. There is no `data-archive` branch anywhere —
any instruction mentioning one is dead.

Every writing workflow does the same three things (`aggregator-update.yml:20,86-93,112-120`;
`price-forecast.yml:58-68,158-167`; `backtest-accuracy.yml:51-61,94-103`):

1. `actions/checkout@v4` of `RayanR000/cs2-oracle-data` at `ref: main`, `path: archive`,
   `fetch-depth: 1`, authenticated with `CS2_DATA_REPO_TOKEN`
2. forecast/backtest additionally `ln -s $PWD/archive/price-archive price-archive`
3. publish with `git checkout --orphan flat && git add -A && git commit && git push --force origin HEAD:main`

**Consequence for debugging: the archive has no git history.** Each publish force-pushes a
single orphan commit, so `git log` will only ever show one. You cannot diff yesterday's
archive against today's, and "a new commit per aggregator run" is not a check you can make.
Verify the archive by its *contents*, not its commits.

Check the last publish:

```bash
gh api repos/RayanR000/cs2-oracle-data/commits/main \
  --jq '.commit.message + "  " + .commit.committer.date'
```

The commit message names the writer and day (`archive: prices + exchange-rates <date>`,
`archive: item_forecasts <date>`, `archive: forecast outcomes + accuracy <date>`), so it
tells you which of the three workflows published last.

Inspect the data itself:

```bash
git clone --depth 1 https://github.com/RayanR000/cs2-oracle-data.git /tmp/cs2-oracle-data
ln -s /tmp/cs2-oracle-data/price-archive "$PWD/price-archive"   # repo root; gitignored
```

`price-archive/` is gitignored (`.gitignore:64-65`) and is expected to be a local checkout
or symlink of that repo — same shape the workflows build. Then, from `backend/`:

```bash
./venv/bin/python -c "
import duckdb
print(duckdb.sql(\"select max(day) latest_day, count(*) rows from '../price-archive/prices-2026-*.parquet'\"))"
```

Layout: `prices-YYYY-MM.parquet` and `snapshots-YYYY-MM.parquet` monthly for the current
year (yearly only for older years), plus `exchange-rates-YYYY.parquet`,
`player-counts-YYYY.parquet`, and a whole `price-archive/ops/` operational layer
(`item_forecasts`, `forecast_outcomes`, `prediction_accuracy`, `collection_runs`,
`accuracy_alerts`, `events`, `event_impacts_denorm`, `supply_snapshots`). The API reads
`ops/` Parquet **before** the DB, which is why a missing mirror write serves stale data
even when Supabase is current.

## Expected patterns

### Healthy state

- Aggregator runs once daily at ~23:00 UTC, writing ~11 distinct source labels
- `prices-2026-*.parquet` totals 11,326,632 rows across 8 monthly files (~101 MB+;
  `prices-2026-07.parquet` alone is 56 MB), growing ~362,586 OHLCV rows/day
- Forecast chains off the aggregator automatically. A retrain costs **176.7s warm /
  250.1s cold** — ~84% of a warm retrain is conformal CV, not booster fitting. Older
  figures (~17 min predict-only, ~53 min Monday) predate the model collapse to 8 models
  and are meaningless now.
- **Monday `mode=full` does NOT guarantee a retrain.** `full` trains only if the model is
  ≥14 days old (`RETRAIN_INTERVAL_DAYS`, `forecast_prices.py:210`) or `FORCE_RETRAIN=1`.
  A fresh model plus `mode=full` predicts and exits. Drift is report-only unless
  `ALLOW_DRIFT_RETRAIN=1`.
- `SKIP_CV=1` is deliberately not set in CI (`price-forecast.yml:119-124`) — it biases the
  conformal `q_hat` low and the served band under-covers. Never add it to buy CI minutes.
- Backtest chains off forecast automatically, ~1-2 min
- A/B regime comparison: `python scripts/forecast_prices.py --compare-regime` (writes
  `lgbm-v3-regime` + `lgbm-v3-global-only` forecasts, runs backtest)
- All tables (`item_forecasts`, `prediction_accuracy`, etc.) stay bounded by UPSERT

### Hang protection: there is only one

**`timeout-minutes: 180` on the `price-forecast` job (`price-forecast.yml:36`) is the only
hang protection that exists.** There are no code-level timeouts: no `TIMEOUT` constant in
`backend/models/`, and `study.optimize()` passes no `timeout=`. Training is fully
sequential, so there is no ensemble or horizon-pool timeout either.

Practically: a wedged forecast run burns three hours and then dies with no diagnostic
beyond the last log line. Read `forecast.log` in the run artifact for the last horizon it
entered. Do not go looking for an Optuna or ensemble timeout to explain a hang — nothing
will have fired.

### Warning signs

- ❌ Frequent failures — check the auto-created issues
- ⏳ Runs missing at expected times — GitHub Actions may be degraded
- Aggregator collecting 0 items — likely CSGOTrader upstream issue
- Accuracy tables not growing — forecast job may have failed; check logs
- **A green badge is not evidence of collection.** Three workflows reported success for
  weeks while storing nothing (see the deletion note above). Audit by querying output-table
  freshness, not by reading Actions. `run_task.py` now fails on zero rows for every count
  field a task returns (`run_task.py:160-186`), but only tasks that go through it are covered.

## Troubleshooting

### Before you diagnose anything: is the fix actually pushed?

A CI-affecting fix only counts once it is on the remote. This has bitten twice — a fix sat
committed-but-unpushed locally while runs kept failing on the old SHA, and the failure was
re-diagnosed from scratch both times. Check first:

```bash
git status -sb                      # look for [ahead N]
git rev-parse HEAD origin/main
gh run list --limit 5 --json name,conclusion,headSha,createdAt
```

If the failing run's `headSha` predates your fix, push and re-run before investigating.

### Workflow didn't run

- Check GitHub Actions status page
- Verify `SUPABASE_DATABASE_URL` and `CS2_DATA_REPO_TOKEN` are set in repository secrets
- Forecast/backtest chain off the upstream workflow — if upstream failed, downstream won't run

### Workflow failed

1. Check the auto-created issue (title includes the workflow name and date)
2. Download the logs artifact from the run
3. Common issues:
   - **`Checkout data archive` / `Checkout price archive` fails** — `CS2_DATA_REPO_TOKEN`
     is missing or expired. Nothing downstream ran.
   - **`alembic upgrade head` fails** — schema drift; run manually against Supabase
   - **CSGOTrader API down** — aggregator returns 0 prices; check upstream
   - **Disk space** — the Parquet steps can grow the checkout on the runner
   - **Out of memory** — the predict phase is chunked (`PREDICT_CHUNK_ITEMS`, default
     1000, `0` disables); lower it before touching the training window
   - **`No boosters restored from the Actions cache`** — see below

### Production backtest failed

Start from the data, not the code. **Production backtest failures are usually archive data
gaps, not code bugs** — see the unresolvable-rate gate below. Confirm archive coverage and
per-day row counts around the forecast dates in play before reading any diff.

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
retrain and repopulate. `mode=train-only` repopulates without touching predict. Note that
`mode=full` retrains only under the age/`FORCE_RETRAIN` conditions above — if the cache is
empty but the model metadata is fresh, use `FORCE_RETRAIN=1` or `mode=train-only`.

### Data not saving

- Verify `SUPABASE_DATABASE_URL` is correct
- Check `alembic current` matches the latest migration (head is **0020**)
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

Anything that touches the archive expects `price-archive/` to exist at the repo root —
create the symlink from the [archive section](#check-the-parquet-archive) first.

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
  5,542-forecast cohort. Check per-day counts around the forecast dates in play:

  ```bash
  ./venv/bin/python -c "
  import duckdb
  print(duckdb.sql(\"select day, count(*) from '../price-archive/prices-2026-*.parquet' \
  where day >= '2026-07-15' group by day order by day\"))"
  ```
- **Archive lag.** Maturity is capped at the archive's last day, so lag alone
  should not trip the gate — the run logs `Archive covers through <date> ... (Nd
  of lag)` when it clamps. If that line shows a large lag, the aggregator is
  behind.

**Rehearsing against SQLite.** `alembic upgrade` cannot replay from scratch on
SQLite: revisions `0001`→`0020` contain Postgres-only `ALTER COLUMN ... TYPE`
DDL. Use `alembic stamp 0020` on the snapshot first, then upgrade. Note that a
rehearsal still rewrites `price-archive/ops/*.parquet`, which is shared — back
those up first.

The collectors for the two deleted workflows still run locally, where residential IPs are
not blocked: `python scripts/run_supply_scraper.py` and
`python scripts/run_task.py reddit_social`. Neither feeds a live feature.

Note: a bare `pytest` from `backend/` aborts during collection —
`scripts/test_social_signal.py:25` is a one-off analysis script (not a test) that imports
`thefuzz`, which is not in `requirements.txt`. Run **`pytest tests`** (714 collected)
until it is renamed or the import is guarded.
