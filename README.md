<p align="center">
  <img src="docs/assets/cs2-oracle-logo-mark.png" width="140" alt="CS2 Oracle logo">
</p>

<h1 align="center">CS2 Oracle</h1>

<p align="center">
  Calibrated price-range forecasts for the Counter-Strike 2 skin market.
</p>

<p align="center">
  <a href="docs/PORTFOLIO.md">Write-up</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#documentation">Docs</a>
</p>

<p align="center">
  <a href="https://github.com/RayanR000/cs2-oracle/actions/workflows/price-forecast.yml"><img src="https://img.shields.io/github/actions/workflow/status/RayanR000/cs2-oracle/price-forecast.yml?label=forecast&style=flat-square&logo=github" alt="Forecast"></a>
  <a href="https://github.com/RayanR000/cs2-oracle/actions/workflows/backtest-accuracy.yml"><img src="https://img.shields.io/github/actions/workflow/status/RayanR000/cs2-oracle/backtest-accuracy.yml?label=backtest&style=flat-square&logo=github" alt="Backtest"></a>
  <img src="https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white&style=flat-square" alt="Python 3.11+">
</p>

CS2 skins trade across a dozen marketplaces with no consolidated price feed. CS2 Oracle
aggregates seven of them into one daily price per item. For every item it forecasts a
calibrated price range at 3, 7, 14 and 30 days.

It runs unattended on GitHub Actions. Most of the engineering went into the evaluation
harness, because on this problem an impressive-looking number is easy to produce and
usually wrong.

**→ [Project write-up](docs/PORTFOLIO.md)**: what was built, what was found, and the
refuted experiments behind it.

**Stack:** Python · LightGBM · split-conformal prediction · DuckDB over Parquet ·
PostgreSQL (Supabase) · FastAPI · GitHub Actions

## What it serves

| Output | Meaning |
|--------|---------|
| **Price band** | Split-conformal interval targeting 80% coverage |
| **Centre** | The band's midpoint |
| `move_odds` | Calibrated probability that the move clears round-trip trading costs |
| `anomaly_p` | Calibrated probability of an abnormal move (3/7/14d only) |

Directional accuracy is tracked but not claimed. The market trends for long enough that a
constant up or down call can look skilful while carrying no information.

## How it works

```
7 market feeds ──► Aggregator ──► Parquet archive ──► Train ──► Calibrate ──► FastAPI
(CSGOTrader dumps)  outlier-voted   (DuckDB, 2013–)    LightGBM   conformal     (Supabase)
                    median                             per horizon band + isotonic
                                                                     │
                                         Backtest ◄──────────────────┘
                                         resolves every forecast against the archive
```

The daily chain is **Aggregator → Price Forecast → Backtest**. Each step triggers when the
previous one succeeds. Freshness and schema-drift checks run alongside it.

| Stage | Detail |
|-------|--------|
| **Aggregate** | Source-quality-weighted median; >2σ quotes rejected, 3-source minimum |
| **Train** | LightGBM per horizon: return regressor, direction classifier, exceedance and anomaly heads. Inputs are price technicals only; 85 non-price features added no measurable accuracy in a 7-fold ablation |
| **Calibrate** | Signed split-conformal band scaled by per-item climatology; isotonic calibration of the probabilities; served-outcome feedback gated per horizon |
| **Evaluate** | MAE, MAPE, interval score and directional accuracy with a Pesaran–Timmermann test. Challenger centres run in shadow and are scored through the same backtest before any manual promotion |

## Quickstart

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/) and a PostgreSQL database.

```bash
uv sync --extra dev                   # add --extra mlops for MLflow tracking
cp backend/.env.example backend/.env  # set DATABASE_URL, STEAM_API_KEY, SECRET_KEY
cd backend
uv run python scripts/run_task.py migrate
uv run uvicorn main:app --port 8000 --reload
```

<details>
<summary><b>Empty database?</b> Create the Alembic version table first</summary>

Some revision IDs are longer than Alembic's default 32 characters:

```sql
CREATE TABLE alembic_version (
  version_num VARCHAR(255) NOT NULL,
  CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);
```

</details>

### API

| Endpoint | Returns |
|----------|---------|
| `GET /items/{id}/prediction` | An item's band, centre, `move_odds` and `anomaly_p` |
| `GET /items/volatility` | Items ranked by expected swing or `move_odds` |
| `GET /accuracy/summary` | Aggregate backtest metrics, filterable by price tier |
| `GET /monitoring/health` | Service and data-freshness status |

Set `DEBUG=true` in `.env` to serve interactive docs at `/docs`.

### Tests

```bash
uv run pytest            # fast gate; excludes tests marked slow
uv run pytest -m slow    # real-trainer tests, several minutes
```

CI also runs `ruff` and `mypy`.

## Repo layout

```
backend/
  api/          FastAPI routes, schemas, serving policy
  collectors/   Aggregator, backfill, supply and volume scrapers
  models/       Forecaster, conformal, recalibration, data quality, source weights
  backtest/     Price resolution, scoring, directional test, promotion gate
  db/           Parquet store, archive reader, candidate store
  migrations/   Alembic revisions
  scripts/      Task runner and batch entrypoints
  tests/        Pytest suite
docs/           Write-up, architecture, specs and plans, research, changelog, experiment log
.github/        Daily chain, freshness, schema drift, lint
```

The price archive lives in a separate repo,
[`cs2-oracle-data`](https://github.com/RayanR000/cs2-oracle-data). Only CI writes to it.
The local `price-archive/` directory is a gitignored, read-only copy.

## Evaluation protocol

Every prediction is stored with its anchor date and horizon. The backtest resolves it once
the archive covers the target date.

- **One code path.** Both legs of every realised return resolve through `backtest.price_resolution`.
- **Frozen outcomes.** Resolved results are never overwritten. Re-resolution is a separate, explicit step.
- **Coverage-bounded maturity.** A forecast matures when the archive covers its target date. Permanent gaps are marked unresolvable so they can't block later runs.
- **Tiered reporting.** Items at $1 and above are the headline cohort, because sub-$1 items behave differently.
- **Baselined direction.** Directional accuracy is always reported alongside the constant-call baseline and a date-clustered Pesaran–Timmermann test.
- **Preregistration.** Experiments commit their hypothesis, metric, pass bars and minimum sample before the data exists, and every harness includes a placebo arm.

Live figures are served at `GET /accuracy/summary`. No headline accuracy is quoted here
until the served series has the ≥20 distinct forecast dates the test needs.

## Documentation

- [`docs/PORTFOLIO.md`](docs/PORTFOLIO.md): project write-up
- [`docs/`](docs/README.md): architecture notes and index
- [`docs/changelog/`](docs/changelog/): 200+ dated decision records with measured effects
- [`docs/research/`](docs/research/): preregistrations and research notes
- [`docs/experiment_log.csv`](docs/experiment_log.csv): every shipped, refuted and void experiment, negative results included
