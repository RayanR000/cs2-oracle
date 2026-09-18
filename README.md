# CS2 Oracle

Daily price forecasts for the Counter-Strike 2 skin market — a LightGBM pipeline that
predicts 3/7/14/30-day returns across a 13-year price archive, with every forecast
scored against what actually happened.

[![Forecast](https://img.shields.io/github/actions/workflow/status/RayanR000/cs2-oracle/price-forecast.yml?label=forecast&style=flat-square&logo=github)](https://github.com/RayanR000/cs2-oracle/actions/workflows/price-forecast.yml)
[![Backtest](https://img.shields.io/github/actions/workflow/status/RayanR000/cs2-oracle/backtest-accuracy.yml?label=backtest&style=flat-square&logo=github)](https://github.com/RayanR000/cs2-oracle/actions/workflows/backtest-accuracy.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white&style=flat-square)](https://python.org)

## What this is

CS2 skins trade across a dozen marketplaces with no consolidated tape. CS2 Oracle is a
range forecaster for that market: per item and horizon it serves a calibrated price
range, its center, and the probability the move clears round-trip cost (`move_odds`).
Directional accuracy is tracked but never a product claim — the market spends long
stretches trending one way, so a constant call can look skilful while carrying no
information.

A daily job pulls seven market price feeds, votes them into one consensus price per
item, appends to a Parquet archive back to 2013, trains gradient-boosted models, and
serves predictions through FastAPI. A separate job resolves every past forecast against
the realised price and writes the accuracy back out. It runs unattended on GitHub
Actions; most engineering effort has gone into the evaluation harness, because on this
problem it is easy to produce an impressive-looking number that is wrong.

## How it works

```
  7 market feeds (CSGOTrader public dumps, 23:00 UTC daily)
  steam · skinport · buff163 · csfloat · csmoney · csgotrader · youpin
                          │
                          ▼
        Aggregator — outlier-voted median per item
        (sources >2σ from median rejected; live asks only,
         3-source minimum; trailing-window means excluded)
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
    Parquet archive              PostgreSQL / Supabase
    prices 2013–present          serving layer + metadata
    queried via DuckDB           (yearly → monthly partitions from 2026)
            │
            ▼
    ══════ TRAINING ══════════════════════════════
    LightGBM per horizon (3/7/14/30d):
      q50 return regressor
      3-class directional classifier (down/flat/up)
      binary exceedance head → P(move > cost)
    price-technicals allowlist · regime-switching
    Optuna tuning · expanding-window CV
            │
            ▼
    Signed split-conformal band, scaled by per-item
    climatology (featureless dispersion, default since 2026-08-20)
            │
            ▼
    ══════ SERVING ═══════════════════════════════
    FastAPI — the API is the only surface; no frontend
            │
            ▼
    Backtest — resolves each forecast against the archive
    MAE · MAPE · directional accuracy, by horizon and price tier
```

**Collection.** Feeds are read from CSGOTrader's public daily dumps rather than each
marketplace's API — freshness traded for reliability and rate-limit headroom.

**Storage.** Training reads the Parquet archive directly, keeping it reproducible from
version-controlled files rather than mutable database state. The durable archive lives
in the separate `cs2-oracle-data` repo (CI-only writes); the local `price-archive/`
copy is gitignored reference.

**Modelling.** The feature set is deliberately narrow: a 7-fold ablation found 85
non-price features added no measurable directional accuracy, so training is restricted
to price technicals. The band comes from split-conformal calibration against
out-of-fold residuals, normalised per item so a $5,000 knife and a $1 case don't share
a width. (Dedicated p10/p90 quantile models were measured and removed: 223s of a 381s
budget for 39–48% coverage against an 80% target.)

## Accuracy and evaluation

Forecast accuracy is measured by a scheduled job, not by a number typed into this file.
Every prediction is stored with anchor date and horizon; once the archive covers the
target date, the backtest resolves it. The protocol:

- **Both legs resolve through the same code path** (`backtest.price_resolution`).
  The stored `current_price` is reference only — scoring against it was a real bug
  that let one cohort read 61.76% and 33.74% on two different days.
- **Resolved outcomes are frozen.** Re-resolution is an explicit, separate operation.
- **Maturity is bounded by archive coverage**, not the calendar — scoring against
  `date.today()` admits the archive lag window as guaranteed misses.
- **Results are reported by price tier.** Sub-$1 items dominate by count and behave
  differently; production reports the ≥$1 cohort.
- **CV and production share one cohort definition**, so offline and live numbers compare.
- **Directional accuracy is never quoted alone** — always beside the constant-call
  baseline, the realised down-rate, and the serial-correlation-robust
  Pesaran–Timmermann test.

**No production accuracy headline is quotable yet** — the PT test needs ≥20 distinct
forecast dates and the served series hasn't accumulated them. That's a calendar problem,
and the honest thing is to say so. Live figures: `GET /accuracy/summary`.

## Quickstart

Requires Python 3.11+ and PostgreSQL 14+ (or a Supabase project).

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # set DATABASE_URL, STEAM_API_KEY, SECRET_KEY
python scripts/run_task.py migrate
uvicorn main:app --port 8000 --reload
```

Key endpoints: `GET /items/{id}/prediction` · `GET /items/volatility` ·
`GET /accuracy/summary`. There is no frontend (removed 2026-08-10, pending rebuild).

Tests: `cd backend && source venv/bin/activate && pytest tests/ -q` (~190 modules).

## Repo layout

```
backend/
  api/routes/        FastAPI handlers (items, market, accuracy, events, …)
  collectors/        Aggregator, supply scraper, validation
  models/
    forecaster.py    Training, features, regimes, CV — the core
    conformal.py     Split-conformal band calibration
  backtest/          Price resolution, scoring, resolution gate
  db/                Parquet store and ops-table mirrors
  scripts/           Task runner and scheduled entrypoints
  tests/             Pytest suite (~190 modules)
price-archive/       Gitignored local copy of the price data (see above)
docs/                Architecture, research, changelog, design specs
.github/workflows/   10 jobs — daily chain Aggregator → Forecast → Backtest,
                     plus freshness check, diagnostics, lint, schema drift
```

Operational reference — env vars, task commands, workflow schedules — lives in
[`docs/`](docs/README.md) (`operations.md`, `architecture/pipeline.md`).

## Limitations

- **Small training universe.** A $1 median-price floor and 1.2M-row budget leave ~900
  of ~5,500 items. Chosen for determinism (the old subsample's seed alone moved CV
  accuracy 1.5–3.1pp); the cost is breadth, and new items can't onboard (discovery
  disabled, backfill quota exhausted).
- **Naive baseline still wins.** Excluding trailing-window means from the consensus
  (2026-08-09) cleaned the label but didn't close the gap to `−return_1d` at any horizon.
- **Walk-forward gate isn't production-comparable.** Its loader skips multi-source
  voting — a relative gate for config changes, not a live-accuracy estimate.
- **Archive day gaps**, mostly from cron drift around midnight UTC, thin training and
  bound what the backtest can resolve.
- **Single-seed models.** A 3-seed ensemble (≈0.3–0.5pp est.) was cut as below what the
  accuracy gate can resolve.

## Documentation

[`docs/`](docs/README.md) holds architecture notes, the research log, preregistered
experiments with recorded outcomes, and an append-only changelog of shipped changes
with measured effects. Negative results are kept — feature families that were built,
measured, and removed explain why.
