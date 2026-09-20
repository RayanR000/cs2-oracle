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

---

CS2 skins trade across a dozen marketplaces with no consolidated tape. CS2 Oracle is a
**range forecaster** for that market: per item and horizon it serves a calibrated price
band, its centre, and the probability the move clears round-trip friction (`move_odds`).
Directional accuracy is tracked but is never the product claim — the market trends long
enough that a constant call looks skilful while carrying no information.

The system runs unattended on GitHub Actions. Most engineering effort has gone into the
evaluation harness, because on this problem it is easy to produce an impressive-looking
number that is wrong.

## Architecture

```
 ┌─────────────────────────────────────────────────────────────────┐
 │  DATA COLLECTION                     23:00 UTC daily            │
 │                                                                 │
 │  7 market feeds via CSGOTrader public dumps                     │
 │  steam · skinport · buff163 · csfloat · csmoney · csgotrader   │
 │  youpin                                                         │
 │       │                                                         │
 │       ▼                                                         │
 │  Aggregator ─── outlier-voted median, source-quality weighted   │
 │  (>2σ rejected · 3-source minimum · trailing means excluded)    │
 └───────┬─────────────────────────────────┬───────────────────────┘
         │                                 │
         ▼                                 ▼
 ┌───────────────────┐           ┌────────────────────┐
 │  Parquet archive   │           │  Supabase / PG     │
 │  2013–present      │           │  serving layer     │
 │  queried via       │           │  monthly parts     │
 │  DuckDB            │           │  from 2026         │
 └───────┬────────────┘           └────────────────────┘
         │
         ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  TRAINING                            per horizon (3/7/14/30d)   │
 │                                                                 │
 │  Champion: LightGBM                                             │
 │    ├─ q50 return regressor                                      │
 │    ├─ 3-class directional classifier                            │
 │    └─ binary exceedance head ─ P(move > cost)                   │
 │                                                                 │
 │  Challengers (shadow):                                          │
 │    ├─ NGBoost distributional  (h=7, h=14)                       │
 │    └─ MAPIE CQR conformal                                       │
 │                                                                 │
 │  price-technicals only · regime-switching · Optuna CV           │
 │  data-quality scoring · MLflow experiment tracking              │
 └───────┬─────────────────────────────────────────────────────────┘
         │
         ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  CALIBRATION                                                    │
 │                                                                 │
 │  Signed split-conformal band, scaled by per-item climatology    │
 │  (featureless dispersion — shipped 2026-08-20)                  │
 └───────┬─────────────────────────────────────────────────────────┘
         │
         ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  SERVING            FastAPI (API-only, no frontend)             │
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

### Design decisions

| Decision | Rationale |
|----------|-----------|
| **Price technicals only** | 7-fold ablation: 85 non-price features added zero measurable accuracy |
| **Per-item climatology band** | 4 sigma-based denominators measured; only featureless dispersion won |
| **Shadow challengers** | NGBoost + CQR run alongside champion; predictions frozen and scored through the same backtest before any promotion |
| **Source-quality weighting** | Per-marketplace reliability weights in the aggregator vote |
| **No quoted accuracy headline** | PT test needs ≥20 distinct forecast dates; the served series hasn't accumulated them yet |

## Quickstart

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # DATABASE_URL, STEAM_API_KEY, SECRET_KEY
python scripts/run_task.py migrate
uvicorn main:app --port 8000 --reload
```

> **Endpoints** &ensp; `GET /items/{id}/prediction` · `GET /items/volatility` · `GET /accuracy/summary`
>
> **Tests** &ensp; `pytest tests/ -q` — 221 modules, ~2,600 tests

## Repo layout

```
backend/
  api/routes/          FastAPI handlers (items, market, accuracy, monitoring, …)
  collectors/          Aggregator, supply scraper, source-quality weighting
  models/              Forecaster, conformal, NGBoost, CQR, centre policy,
                       data quality, source weights, MLflow, forecast assembly
  backtest/            Price resolution, scoring, interval metrics,
                       candidate resolution + scoring, promotion gate
  db/                  Parquet store, ops-table mirrors
  scripts/             Task runner, ~47 batch entrypoints
  tests/               Pytest suite (221 modules)
price-archive/         Gitignored local copy of the durable archive
docs/                  Architecture, research log, experiment log, changelog
.github/workflows/     10 workflows — daily chain + freshness, lint, diagnostics
```

> The durable price archive lives in the separate [`cs2-oracle-data`](https://github.com/RayanR000/cs2-oracle-data) repo (CI-only writes, orphan-commit + force-push). The local `price-archive/` is read-only reference.

## Evaluation protocol

Every prediction is stored with anchor date and horizon; the backtest resolves it once
the archive covers the target date.

- Both legs resolve through one code path (`backtest.price_resolution`)
- Resolved outcomes are frozen — re-resolution is explicit and separate
- Maturity is bounded by archive coverage, not the calendar
- Results reported by price tier (≥$1 cohort; sub-$1 items behave differently)
- CV and production share one cohort definition
- Directional accuracy is never quoted alone — always beside the constant-call baseline, realised down-rate, and the Pesaran–Timmermann test

Live figures: `GET /accuracy/summary`.

## Limitations

| Limitation | Detail |
|------------|--------|
| **Small training universe** | $1 floor + 1.2M-row budget → ~900 of ~5,500 items; discovery disabled |
| **Naive baseline competitive** | Clean consensus label didn't close the gap to `-return_1d` |
| **Walk-forward ≠ production** | Loader skips multi-source voting; relative gate only |
| **Archive day gaps** | Cron drift around midnight UTC thins training windows |
| **Single-seed models** | 3-seed ensemble cut — delta below what the accuracy gate resolves |
| **Manual promotion** | Shadow challengers produce evidence reports; promotion requires operator approval |

## Documentation

[`docs/`](docs/README.md) — architecture notes, research log, preregistered experiments with
recorded outcomes, and an append-only changelog of shipped changes with measured effects.
[`experiment_log.csv`](docs/experiment_log.csv) tracks every shipped / refuted / void arm.
Negative results are kept.
