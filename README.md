# CS2 Oracle

Daily price forecasts for the Counter-Strike 2 skin market — a LightGBM pipeline that
predicts 3/7/14/30-day returns across a 13-year price archive, with every forecast
scored against what actually happened.

[![Forecast](https://img.shields.io/github/actions/workflow/status/RayanR000/cs2-oracle/price-forecast.yml?label=forecast&style=flat-square&logo=github)](https://github.com/RayanR000/cs2-oracle/actions/workflows/price-forecast.yml)
[![Backtest](https://img.shields.io/github/actions/workflow/status/RayanR000/cs2-oracle/backtest-accuracy.yml?label=backtest&style=flat-square&logo=github)](https://github.com/RayanR000/cs2-oracle/actions/workflows/backtest-accuracy.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white&style=flat-square)](https://python.org)

## What this is

CS2 skins trade across a dozen marketplaces with no consolidated tape. Prices diverge
between venues, listings are thin, and the public "analytics" sites mostly show you a
line chart of where a price has already been.

CS2 Oracle is an attempt at the harder version: forecasting where a price is going, and
then being honest about how often that forecast was right. A daily job pulls seven market
price feeds, votes them into a single consensus price per item, appends to a Parquet
archive going back to 2013, trains gradient-boosted models on the result, and serves
predictions through a FastAPI backend. A separate scheduled job resolves every past
forecast against the realised price and writes the accuracy back out.

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
    LightGBM, 4 horizons (3/7/14/30d)
    q50 regressor + directional classifier per horizon
    regime-switching · Optuna · expanding-window CV
            │
            ▼
    Conformal calibration → uncertainty band
            │
            ▼
    ══════ SERVING ═══════════════════════════════════
    FastAPI — the API is the only surface; there is no frontend
            │
            ▼
    Backtest — resolves each forecast against the archive
    MAE · MAPE · directional accuracy, by horizon and price tier
```

**Collection.** Seven marketplaces are read from CSGOTrader's public daily dumps rather
than from each marketplace's own API — a deliberate tradeoff of freshness for reliability
and rate-limit headroom. Sources are reconciled per item by outlier-voted median: with
three or more sources on an item-day, anything more than 2σ from the median is rejected
and the median of the rest becomes the consensus price. Only live *ask* prices vote —
BUFF's `highest_order` is a bid, and Steam's 7/30/90-day feeds are trailing-window mean
sale prices; all four are dropped before the group is read, so they count toward neither
the median nor the three-source gate.

**Storage.** The archive is Parquet on disk, partitioned yearly through 2025 and monthly
from 2026, queried with DuckDB. Training reads the archive directly; the database holds
only the serving layer and item metadata. This keeps the training set reproducible from
version-controlled files rather than from mutable database state.

**Modelling.** Per horizon: one LightGBM median regressor for the return, and one 3-class
directional classifier (down / flat / up) that supplies the served direction. Regime
models (bear / range / bull, split on 30-day market return at ±3%) are fit for the
prevailing regime and fall back to the global model otherwise. Hyperparameters are tuned
with Optuna per horizon.

The feature set is deliberately narrow. A global allowlist restricts training to price
technicals; a 7-fold ablation found the 85 non-price features added no measurable
directional accuracy over price and technical features alone, and hurt at 3d and 30d.
Cross-sectional features are excluded again at 14d and 30d, and event features at 30d.

**Uncertainty.** The prediction band comes from split-conformal calibration against
out-of-fold residuals, not from quantile models. The nonconformity score is normalised by
a per-item denominator so a $5,000 knife and a $1 case do not get the same width; since
2026-08-19 that denominator is a featureless per-item climatology rather than a modelled
sigma (`CLIMATOLOGY_SCALE`, on by default). Dedicated p10/p90 GBMs were measured and
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
- **Directional accuracy is never quoted on its own.** This market spends long stretches
  trending one way, so a constant "always down" call can post a high hit rate while
  carrying no information. The headline is a serial-correlation-robust
  Pesaran–Timmermann test; DA is reported only alongside the constant-call baseline and
  the realised down-rate.

**There is no quotable production accuracy figure yet.** The Pesaran–Timmermann test
requires at least 20 distinct forecast dates, and the served series does not yet cover
that many, so every horizon currently reports no headline. This is a calendar problem
rather than a code problem, and the honest thing is to say so instead of publishing the
blended number that is available.

Live figures are served at `GET /accuracy/summary`.

## Quickstart

Requires Python 3.11+ and PostgreSQL 14+ (or a Supabase project).

```bash
# Backend
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # set DATABASE_URL, STEAM_API_KEY, SECRET_KEY
python scripts/run_task.py migrate
uvicorn main:app --port 8000 --reload
```

There is no frontend. It was deleted on 2026-08-10 to be rebuilt from scratch — see
[`docs/changelog/2026-08-10-frontend-removed.md`](docs/changelog/2026-08-10-frontend-removed.md).
The API is the product surface until then.

Tests: `cd backend && source venv/bin/activate && pytest tests/ -q`.

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
  tests/             Pytest suite (146 modules, 2,464 tests)
price-archive/       Parquet price data, 2013-present
docs/                Architecture, research, changelog, design specs
.github/workflows/   9 jobs: 5 cron, 1 chained, 2 manual, 1 on PR/push
```

Operational reference — API endpoints, environment variables, task commands, workflow
schedules — lives in [`docs/`](docs/README.md).

## Limitations

Known weaknesses, stated plainly:

- **The training universe is small.** Training applies a $1 median-price floor and a
  1.2M feature-row budget, which the ≥$1 cohort — 916 of 5,536 items — fits whole, so no
  item subsample is drawn. That was chosen for determinism, not accuracy: the previous
  subsample's seed alone moved cross-validated accuracy by 1.5–3.1pp, which is larger
  than most effects being measured. The cost is that the model sees under a thousand
  items, and nothing can currently add more (see below).
- **Some sources are not point observations.** Steam's 7/30/90-day feeds are
  trailing-window *mean sale* prices, and they used to vote in the consensus alongside
  live ask prices, smearing the series on illiquid items into frozen runs. They were
  excluded from the vote on 2026-08-09 (`TRAILING_WINDOW_SOURCES`), and the model was
  retrained on the corrected consensus the next day. The correction cleaned the label's
  basis but did not close the gap to the naive `−return_1d` baseline, which still wins at
  every horizon.
- **The walk-forward gate is not directly comparable to production.**
  `walkforward_backtest.py` uses its own price loader that skips multi-source voting and
  the backfill filter, collapsing duplicate item-days with a plain mean where production
  takes an outlier-voted median. It scores a different price consensus over a different
  item universe than production trains on. It is a relative gate for config changes, not
  an estimate of live accuracy.
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
