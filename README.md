# CS2 Oracle

Daily price forecasts for the Counter-Strike 2 skin market — a LightGBM pipeline that
predicts 3/7/14/30-day returns across a 13-year price archive, with every forecast
scored against what actually happened.

[![Forecast](https://img.shields.io/github/actions/workflow/status/RayanR000/cs2-oracle/price-forecast.yml?label=forecast&style=flat-square&logo=github)](https://github.com/RayanR000/cs2-oracle/actions/workflows/price-forecast.yml)
[![Backtest](https://img.shields.io/github/actions/workflow/status/RayanR000/cs2-oracle/backtest-accuracy.yml?label=backtest&style=flat-square&logo=github)](https://github.com/RayanR000/cs2-oracle/actions/workflows/backtest-accuracy.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white&style=flat-square)](https://python.org)

<!-- Add a dashboard screenshot here — one image, ideally the item detail page with a
     forecast overlay. This is the highest-value single addition to this README. -->

## What this is

CS2 skins trade across a dozen marketplaces with no consolidated tape. Prices diverge
between venues, listings are thin, and the public "analytics" sites mostly show you a
line chart of where a price has already been.

CS2 Oracle is an attempt at the harder version: forecasting where a price is going, and
then being honest about how often that forecast was right. A daily job pulls seven market
price feeds, votes them into a single consensus price per item, appends to a Parquet
archive going back to 2013, trains gradient-boosted models on the result, and serves
predictions through a FastAPI backend to a Next.js dashboard. A separate scheduled job
resolves every past forecast against the realised price and writes the accuracy back out.

It is a single-operator system that runs unattended on GitHub Actions. Most of the
engineering effort has gone into the evaluation harness rather than the model, because on
this problem it is very easy to produce an impressive-looking number that is wrong.

## How it works

```
  7 market price feeds (CSGOTrader public dumps)
  steam · skinport · buff163 · csfloat · csmoney · csgotrader · youpin
                          │
                          ▼
        Daily aggregator — 23:00 UTC
        outlier-voted median, sources >2σ from median rejected
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
    Parquet archive              PostgreSQL / Supabase
    prices 2013-2026             serving layer, daily closes
    queried via DuckDB
            │
            ▼
    ══════ TRAINING ══════════════════════════════════
    LightGBM, 4 horizons (3/7/14/30d), median regression
    regime-switching · Optuna · expanding-window CV
            │
            ▼
    Conformal calibration → uncertainty band
            │
            ▼
    ══════ SERVING ═══════════════════════════════════
    FastAPI  →  Next.js dashboard
            │
            ▼
    Backtest — resolves each forecast against the archive
    MAE · MAPE · directional accuracy, by horizon and price tier
```

**Collection.** Seven price feeds are read from CSGOTrader's public daily dumps rather
than from each marketplace's own API — a deliberate tradeoff of freshness for reliability
and rate-limit headroom. Sources are reconciled per item by outlier-voted median, so a
single stale or mispriced venue cannot move the consensus.

**Storage.** The archive is Parquet on disk, partitioned yearly through 2025 and monthly
from 2026, queried with DuckDB. Training reads the archive directly; the database holds
only the serving layer and item metadata. This keeps the training set reproducible from
version-controlled files rather than from mutable database state.

**Modelling.** One LightGBM median regressor per horizon, with separate models per market
regime (bear / range / bull, split on 30-day market return at ±3%). Hyperparameters are
tuned with Optuna per horizon. Feature groups are excluded per horizon based on an
ablation study — cross-sectional features measurably *hurt* the 14d and 30d horizons, so
they are not used there.

**Uncertainty.** The prediction band comes from split-conformal calibration against
out-of-fold residuals, not from quantile models. Dedicated p10/p90 GBMs were measured and
removed: they consumed 223s of a 381s training budget to deliver 39–48% empirical
coverage against an 80% target.

**Evaluation.** See below — it's the part worth reading.

## Accuracy and evaluation

Forecast accuracy is measured by a scheduled job, not by a number typed into this file.
Every prediction the system serves is written with its anchor date and horizon; once the
archive covers the target date, the backtest resolves it and records the outcome.

The protocol, and the reasons for it:

- **Both legs resolve through the same code path.** The base price and the actual price
  are both resolved by `backtest.price_resolution.resolve_anchors`. The stored
  `current_price` on a forecast is kept for reference and is never scored against — using
  it as the base leg is a real bug this system had, and it let one cohort score 61.76% and
  33.74% on two different days.
- **Resolved outcomes are frozen.** Once scored, `base_price` / `actual_price` /
  `resolved_at` are final. Re-resolution is an explicit, separate operation.
- **Maturity is bounded by archive coverage, not by the calendar.** A forecast is only
  evaluable once the archive actually covers its target date. Scoring against
  `date.today()` admits the archive's lag window and puts guaranteed misses into the
  cohort.
- **Results are reported by price tier.** Sub-$1 items dominate by count and behave
  differently from the rest of the market, so a single blended accuracy figure is mostly
  a statement about penny items. Production reports the ≥$1 cohort.
- **Cross-validation and production are scored on the same cohort definition**, so the
  offline number and the live number are comparable.

<!-- Accuracy table goes here. Report per horizon, ≥$1 cohort, against a persistence
     baseline, with n and the evaluation window stated. -->

Live figures are served at `GET /accuracy/summary` and rendered on the dashboard.

## Quickstart

Requires Python 3.11+, Node 20+, and PostgreSQL 14+ (or a Supabase project).

```bash
# Backend
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # set DATABASE_URL, STEAM_API_KEY, SECRET_KEY
python scripts/run_task.py migrate
uvicorn main:app --port 8000 --reload
```

```bash
# Frontend
cd frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
npm run dev                   # → http://localhost:3000
```

Tests: `cd backend && source venv/bin/activate && pytest`

## Repo layout

```
backend/
  api/routes/        FastAPI route handlers
  collectors/        Market aggregator, supply scraper, validation
  models/
    forecaster.py    Training, features, regimes, CV — the core of the project
    conformal.py     Split-conformal band calibration
  backtest/          Price resolution, scoring, resolution gate
  db/                Parquet store and ops-table mirrors
  scripts/           Task runner and scheduled entrypoints
  tests/             Pytest suite (41 modules, 714 tests)
frontend/
  app/               Next.js app router pages
  components/        React components
  lib/               API client
price-archive/       Parquet price data, 2013-present
docs/                Architecture, research, changelog, design specs
.github/workflows/   3 cron jobs + 1 chained + 1 manual
```

Operational reference — API endpoints, environment variables, task commands, workflow
schedules — lives in [`docs/`](docs/README.md).

## Limitations

Known weaknesses, stated plainly:

- **Training runs on a subsample.** The default budget is 100,000 feature rows
  (`TRAIN_FEATURE_ROWS`), a fraction of the available pool. Raising it to 700,000 costs
  4.5× the training wall-clock, and the accuracy gate cannot currently resolve whether
  that buys anything, so it has not been raised.
- **The walk-forward gate is not directly comparable to production.**
  `walkforward_backtest.py` uses its own price loader that skips multi-source voting and
  the backfill filter, so it scores a different price consensus over a different item
  universe than production trains on. It is a relative gate for config changes, not an
  estimate of live accuracy.
- **The archive has day gaps.** Missing days come mostly from cron drift around midnight
  UTC rather than from failed runs, but they thin the training set and bound what the
  backtest can resolve.
- **No path to onboard new items.** The catalog is fixed to backfilled items. Steam
  discovery is disabled and the third-party backfill quota is exhausted, so newly tradable
  skins do not enter the system.
- **Single-member models.** The ensemble is one seed. A 3-seed ensemble was estimated at
  0.3–0.5pp, below what the accuracy gate can resolve, so it was cut rather than kept on
  faith.

## Documentation

[`docs/`](docs/README.md) holds architecture notes, the research log, and a changelog of
shipped changes with their measured effect. Negative results are kept there too — several
feature families were built, measured, and removed, and the write-ups explain why.
