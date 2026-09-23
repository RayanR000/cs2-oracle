<h1 align="center">CS2 Oracle</h1>

<p align="center">
  Calibrated price-range forecasts for the Counter-Strike 2 skin market.
</p>

<p align="center">
  <a href="https://github.com/RayanR000/cs2-oracle/actions/workflows/price-forecast.yml"><img src="https://img.shields.io/github/actions/workflow/status/RayanR000/cs2-oracle/price-forecast.yml?label=forecast&style=flat-square&logo=github" alt="Forecast"></a>
  <a href="https://github.com/RayanR000/cs2-oracle/actions/workflows/backtest-accuracy.yml"><img src="https://img.shields.io/github/actions/workflow/status/RayanR000/cs2-oracle/backtest-accuracy.yml?label=backtest&style=flat-square&logo=github" alt="Backtest"></a>
  <img src="https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white&style=flat-square" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/horizons-3%2F7%2F14%2F30d-blue?style=flat-square" alt="Horizons">
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#evaluation-protocol">Evaluation</a> ·
  <a href="#limitations">Limitations</a> ·
  <a href="docs/PORTFOLIO.md">Write-up</a>
</p>

---

## Overview

CS2 skins trade across a dozen marketplaces with no consolidated tape. CS2 Oracle is a
**range forecaster** for that market. For every item and horizon it serves:

| Output | Meaning |
|--------|---------|
| **Price band** | A split-conformal interval targeting 80% coverage |
| **Centre** | The band's midpoint |
| `move_odds` | Calibrated probability that the move clears round-trip trading costs |
| `anomaly_p` | Calibrated probability of an abnormal move (3/7/14d only) |

The system tracks directional accuracy but doesn't claim it as a result. The market trends
long enough that a constant up or down call can look skilful while carrying no information.

The pipeline runs unattended on GitHub Actions. Most of the engineering went into the
evaluation harness, because on this problem it is easy to produce an impressive-looking
number that turns out to be wrong.

> **→ [Project write-up](docs/PORTFOLIO.md)**: what was built, what was found, and the 33
> refuted experiments behind it.

**Stack:** Python · LightGBM · split-conformal prediction · DuckDB over Parquet ·
PostgreSQL (Supabase) · FastAPI · GitHub Actions · pytest

## Architecture

```
 ┌─────────────────────────────────────────────────────────────────┐
 │  DATA COLLECTION                     23:00 UTC daily            │
 │                                                                 │
 │  7 market feeds via CSGOTrader public dumps                     │
 │  steam · skinport · buff163 · csfloat · csmoney · csgotrader    │
 │  youpin                                                         │
 │       │                                                         │
 │       ▼                                                         │
 │  Aggregator ─── outlier-voted median, source-quality weighted   │
 │  (>2σ rejected · 3-source minimum · trailing means excluded)    │
 └───────┬─────────────────────────────────┬───────────────────────┘
         │                                 │
         ▼                                 ▼
 ┌────────────────────┐           ┌────────────────────┐
 │  Parquet archive   │           │  Supabase / PG     │
 │  2013–present      │           │  serving layer     │
 │  DuckDB queries    │           │                    │
 │  monthly files     │           │                    │
 └───────┬────────────┘           └────────────────────┘
         │
         ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  TRAINING                            per horizon (3/7/14/30d)   │
 │                                                                 │
 │  Champion: LightGBM                                             │
 │    ├─ q50 return regressor                                      │
 │    ├─ 3-class directional classifier (tracked, not claimed)     │
 │    ├─ binary exceedance head ─ P(move > cost)                   │
 │    └─ anomaly head ─ P(abnormal move)                           │
 │                                                                 │
 │  Challengers (shadow): centre candidates, frozen and scored     │
 │    through the same backtest before any promotion               │
 │                                                                 │
 │  price technicals only · Optuna tuning · data-quality scoring   │
 └───────┬─────────────────────────────────────────────────────────┘
         │
         ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  CALIBRATION                                                    │
 │                                                                 │
 │  Signed split-conformal band, scaled by per-item climatology    │
 │  Isotonic calibration of move_odds and anomaly_p                │
 │  Served-outcome feedback, gated per horizon                     │
 └───────┬─────────────────────────────────────────────────────────┘
         │
         ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  SERVING            FastAPI (API only, no frontend)             │
 └───────┬─────────────────────────────────────────────────────────┘
         │
         ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  EVALUATION                                                     │
 │                                                                 │
 │  Backtest: resolves every forecast against the archive          │
 │  MAE · MAPE · interval score · directional accuracy + PT test   │
 │  Champion–challenger promotion gate (manual, evidence-scored)   │
 └─────────────────────────────────────────────────────────────────┘
```

The daily chain is **Aggregator → Price Forecast → Backtest**. Each step is triggered when
the previous one succeeds, and freshness and schema-drift checks run alongside it.

### Design decisions

| Decision | Rationale |
|----------|-----------|
| **Price technicals only** | 7-fold ablation: 85 non-price features added no measurable accuracy |
| **Per-item climatology band** | Four modelled-σ width denominators were measured; only featureless dispersion won |
| **Signed conformal** | Separate lower and upper quantiles replace one symmetric width |
| **Shadow challengers** | Candidate centres run alongside the champion and are scored through the same backtest before any promotion |
| **Source-quality weighting** | Per-marketplace reliability weights in the aggregator vote |
| **No quoted accuracy headline** | The PT test needs ≥20 distinct forecast dates, and the served series hasn't accumulated them yet |

## Quickstart

Dependencies are managed with [uv](https://docs.astral.sh/uv/) from the repo root.

```bash
uv sync --extra dev                   # add --extra mlops for MLflow tracking
cp backend/.env.example backend/.env  # DATABASE_URL, STEAM_API_KEY, SECRET_KEY
cd backend
uv run python scripts/run_task.py migrate
uv run uvicorn main:app --port 8000 --reload
```

> [!NOTE]
> Before migrating an **empty** database, create `alembic_version` with a `VARCHAR(255)`
> `version_num` column. Some revision IDs are longer than Alembic's default 32 characters.
> See `.github/workflows/schema-drift-check.yml` for the exact statement.

### API

The main endpoints:

| Endpoint | Returns |
|----------|---------|
| `GET /items/{id}/prediction` | An item's band, centre, `move_odds` and `anomaly_p` |
| `GET /items/volatility` | Items ranked by expected swing or `move_odds`, with stability tags |
| `GET /accuracy/summary` | Aggregate backtest metrics, filterable by price tier |
| `GET /monitoring/health` | Service and data-freshness status |

With `DEBUG=true` in `.env`, interactive docs are served at `/docs`.

### Tests

```bash
uv run pytest                         # fast gate (excludes tests marked slow)
uv run pytest -m slow                 # real-trainer tests, several minutes
```

The suite has about 2,500 tests across 209 modules. CI also runs `ruff` and `mypy`.

## Repo layout

```
backend/
  api/                 FastAPI routes, schemas, serving policy
  collectors/          Aggregator, CSMarketAPI backfill, supply and volume scrapers
  models/              Forecaster, conformal, served recalibration, data quality,
                       source weights, forecast assembly, MLflow utilities
  backtest/            Price resolution, scoring, directional test,
                       candidate resolution and scoring, promotion gate
  db/                  Parquet store, archive reader, candidate store
  monitoring/          Feature drift
  migrations/          Alembic revisions
  scripts/             Task runner and batch entrypoints (forecast, backtest, archive)
  tests/               Pytest suite
docs/                  Architecture, research, changelog, experiment log
.github/workflows/     Daily chain, freshness, schema drift, lint, diagnostics
```

> [!IMPORTANT]
> The durable price archive lives in the separate
> [`cs2-oracle-data`](https://github.com/RayanR000/cs2-oracle-data) repo. Only CI writes to
> it, using an orphan commit and force-push. The local `price-archive/` directory is a
> gitignored, read-only copy.

## Evaluation protocol

Every prediction is stored with its anchor date and horizon. The backtest resolves it once
the archive covers the target date.

- **One code path.** Both legs of every realised return resolve through `backtest.price_resolution`.
- **Frozen outcomes.** Resolved results are never overwritten; re-resolution is an explicit, separate step.
- **Coverage-bounded maturity.** A forecast matures when the archive covers its target date, not when the calendar says so. Permanent gaps are marked unresolvable so they can't block later runs.
- **Tiered reporting.** Results are split by price tier, with ≥$1 as the headline cohort, because sub-$1 items behave differently.
- **Shared cohorts.** Cross-validation and production use the same cohort definition.
- **Baselined direction.** Directional accuracy is always reported next to the constant-call baseline, the realised down-rate, and a date-clustered Pesaran–Timmermann test. It is never quoted alone.
- **Preregistration.** Experiments commit their hypothesis, metric, pass bars and minimum sample size before the data exists, and every harness includes a placebo arm.

Live figures are served at `GET /accuracy/summary`.

## Limitations

| Limitation | Detail |
|------------|--------|
| **Band over-covers** | Served coverage runs above the 80% target. Correction goes through served-outcome feedback, one horizon at a time as the data matures |
| **No directional or trading edge** | After the market factor is removed, no per-item directional signal remains. A paper-trading audit selects zero trades once costs are applied |
| **Centre adds no skill** | The model centre doesn't beat last price, so the band is the product |
| **Small training universe** | A $1 price floor and a 1.2M-row budget limit training to about 900 of ~5,500 items |
| **Walk-forward ≠ production** | The walk-forward loader skips multi-source voting, so it can only be used as a relative gate |
| **Archive day gaps** | Cron drift around midnight UTC thins training windows |
| **Single-seed models** | A 3-seed ensemble was cut because its gain was below what the accuracy gate can resolve |
| **Manual promotion** | Shadow challengers produce evidence reports, and an operator approves any promotion |

## Documentation

- [`docs/PORTFOLIO.md`](docs/PORTFOLIO.md): project write-up
- [`docs/`](docs/README.md): architecture notes and the documentation index
- [`docs/changelog/`](docs/changelog/): 200+ dated decision records with measured effects
- [`docs/research/`](docs/research/): preregistrations and research notes
- [`docs/experiment_log.csv`](docs/experiment_log.csv): every shipped, refuted and void experiment

Negative results are kept.
