<div align="center">

# CS2 Oracle

**Market intelligence platform for the Counter-Strike 2 skin economy**

[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white&style=flat-square)](https://python.org)
[![Node 20+](https://img.shields.io/badge/node-20%2B-339933?logo=nodedotjs&logoColor=white&style=flat-square)](https://nodejs.org)
[![Next.js 16](https://img.shields.io/badge/next.js-16.2-000000?logo=nextdotjs&logoColor=white&style=flat-square)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/fastapi-009688?logo=fastapi&logoColor=white&style=flat-square)](https://fastapi.tiangolo.com)
[![TypeScript](https://img.shields.io/badge/typescript-3178C6?logo=typescript&logoColor=white&style=flat-square)](https://typescriptlang.org)
[![Tailwind CSS v4](https://img.shields.io/badge/tailwind_v4-06B6D4?logo=tailwindcss&logoColor=white&style=flat-square)](https://tailwindcss.com)
[![LightGBM](https://img.shields.io/badge/lightgbm-7D3C98?logo=python&logoColor=white&style=flat-square)](https://lightgbm.readthedocs.io)

[![Aggregator](https://img.shields.io/github/actions/workflow/status/RayanR000/cs2-oracle/aggregator-update.yml?label=aggregator&style=flat-square&logo=github)](https://github.com/RayanR000/cs2-oracle/actions/workflows/aggregator-update.yml)
[![Forecast](https://img.shields.io/github/actions/workflow/status/RayanR000/cs2-oracle/price-forecast.yml?label=forecast&style=flat-square&logo=github)](https://github.com/RayanR000/cs2-oracle/actions/workflows/price-forecast.yml)
[![Backtest](https://img.shields.io/github/actions/workflow/status/RayanR000/cs2-oracle/backtest-accuracy.yml?label=backtest&style=flat-square&logo=github)](https://github.com/RayanR000/cs2-oracle/actions/workflows/backtest-accuracy.yml)
[![Supply](https://img.shields.io/github/actions/workflow/status/RayanR000/cs2-oracle/supply-scraper.yml?label=supply&style=flat-square&logo=github)](https://github.com/RayanR000/cs2-oracle/actions/workflows/supply-scraper.yml)
[![Sentiment](https://img.shields.io/github/actions/workflow/status/RayanR000/cs2-oracle/reddit-sentiment.yml?label=sentiment&style=flat-square&logo=github)](https://github.com/RayanR000/cs2-oracle/actions/workflows/reddit-sentiment.yml)

</div>

---

**CS2 Oracle** is a full-stack analytics platform for the Counter-Strike 2 skin economy. A daily pipeline collects multi-source prices across **7 markets**, archives 13+ years of history to **Parquet**, and serves intelligence through a **FastAPI** REST API to a **Next.js** dashboard. ML price forecasts via **LightGBM quantile ensembles** replace naive trend analysis with probabilistic predictions.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────┐  │
│  │  Steam   │ │ Skinport │ │  Buff163 │ │  CSFloat  │ │  … +3 │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬───┘  │
│       └────────────┴────────────┴────────────┴────────────┘      │
│                              ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                Daily Aggregator (23:00 UTC)               │    │
│  └──────────┬────────────────────────────────────┬──────────┘    │
│             ▼                                    ▼                │
│  ┌──────────────────┐              ┌──────────────────────┐      │
│  │  Parquet Archive  │              │  PostgreSQL/Supabase  │      │
│  │  (13 yrs · DuckDB)│              │  (daily closes)       │      │
│  └──────────────────┘              └──────────┬───────────┘      │
└─────────────────────────────────────────────────┼────────────────┘
                                                  │
┌─────────────────────────────────────────────────▼────────────────┐
│                           BACKEND                                 │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐  ┌─────────┐  │
│  │  FastAPI   │  │ SQLAlchemy │  │   Alembic    │  │  Pydantic│  │
│  │  REST API  │  │    ORM     │  │  Migrations  │  │ Settings │  │
│  └─────┬──────┘  └────────────┘  └──────────────┘  └─────────┘  │
│        │                                                          │
│  ┌─────▼──────────────────────────────────────────────────────┐  │
│  │  ML Layer: LightGBM · Optuna · Quantile Ensembles         │  │
│  │  Regime-Switching · Walk-Forward · SHAP Feature Importance│  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────┐
│                      FRONTEND                                     │
│  ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌────────────────┐   │
│  │  Next.js   │ │  Recharts  │ │  Tailwind│ │  Framer Motion │   │
│  │  App Router│ │  Viz       │ │  CSS v4  │ │  Animations    │   │
│  └────────────┘ └────────────┘ └──────────┘ └────────────────┘   │
│                                                                   │
│  Item Detail · Price History · Forecast Overlay · Market Map    │
│  Opportunities · Event Impacts · Portfolio · Accuracy Metrics   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Features

| | |
|---|---|
| **Multi-Source Collection** — Daily aggregator polling 7 CS2 skin markets in parallel | **Parquet Price Archive** — 13 years of history, queryable via DuckDB |
| **ML Price Forecasts** — LightGBM quantile regression (q10/q50/q90) across 3/7/14/30d horizons. 6-member diversified ensemble with walk-forward validation & Optuna tuning | **Regime-Switching** — Separate models per market regime (bear / range / bull) |
| **Accuracy Tracking** — Automated daily backtesting with MAE, MAPE, directional accuracy, drift alerts | **Model Explainability** — Per-item feature importance via SHAP-style analysis |
| **Event Impact Analysis** — Quantified market-event price impacts with correlation scoring | **Market Opportunities** — Undervalued, overheated, and momentum signals surfaced daily |
| **Interactive Dashboard** — Responsive Recharts visualizations, grouped market views, item detail pages, forecast overlays | **A/B Testing** — Regime vs. ensemble comparison for forecast methodology validation |

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Next.js 16, React 19, TypeScript, Tailwind CSS 4, Recharts, Framer Motion |
| **Backend** | Python 3.11, FastAPI, SQLAlchemy 2, Alembic, Pydantic Settings |
| **Data** | PostgreSQL / Supabase, DuckDB, Parquet, Pandas, NumPy, SciPy |
| **Machine Learning** | LightGBM, Optuna, Joblib |
| **Automation** | GitHub Actions (7 scheduled workflows) |

---

## Quick Start

### Prerequisites

- Python 3.11+, Node.js 20+, PostgreSQL 14+ (or Supabase account)

### Backend

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # configure DATABASE_URL + API keys
python scripts/run_task.py migrate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
npm run dev                   # → http://localhost:3000
```

---

## API Overview

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /items/`, `/items/search`, `/items/trending`, `/items/{id}` | Item catalog & detail |
| `GET /items/{id}/price-history`, `/prediction` | Price timeline + ML forecasts |
| `GET /market/summary` | Grouped market view (paginated, cached) |
| `GET /opportunities/`, `/undervalued`, `/overheated`, `/momentum` | Trading signals |
| `GET /events/`, `/events/recent` | Market event timeline |
| `GET /accuracy/`, `/accuracy/latest`, `/accuracy/summary` | Forecast accuracy metrics |
| `GET /ab-test/regime`, `/ab-test/ensemble` | Forecast methodology A/B comparison |
| `GET /auth/me`, `/auth/steam/login` | Steam OpenID authentication |
| `GET /portfolio/inventory` | Steam inventory snapshot |

---

## Environment Variables

Key configuration (see `backend/.env.example` for full reference):

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL/Supabase connection string |
| `SECRET_KEY` | Yes | Session signing key (rotate in production) |
| `STEAM_API_KEY` | Yes | Steam Web API key |
| `CSMARKETAPI_*` | Yes | Market API keys (6 sources) |
| `FRONTEND_URL` | Yes | Allowed CORS origin |
| `ENVIRONMENT` | No | `development` / `production` |

---

## Commands

### Backend Tasks

Run via `python scripts/run_task.py <task>`:

| Task | Description |
|------|-------------|
| `aggregate` | Full aggregator collection (all items, all sources) |
| `priority` | Top 2000 items collection |
| `supply_scrape` | Listing supply-depth collection |
| `migrate` | Run pending Alembic migrations |
| `backtest` | Run forecast accuracy backtest |
| `backtest_historical` | Run historical walk-forward backtest |
| `walkforward_report` | Generate walk-forward validation report |
| `event_correlation` | Run market-event correlation analysis |
| `reddit_social` | Run Reddit sentiment analysis |

### Frontend

| Command | Description |
|---------|-------------|
| `npm run dev` | Development server (localhost:3000) |
| `npm run build` | Type-check + production build |
| `npm run lint` | ESLint |

### Testing

```bash
cd backend && source venv/bin/activate && pytest
```

---

## Data Pipeline

```
22:00 UTC     Supply Scraper        → listing supply depth        → Supabase
23:00 UTC     Aggregator            → 7 source prices             → Parquet + Supabase
Every 2h      Player Count          → Steam active players        → Supabase + Parquet
(chained)     Forecast              → LightGBM ensemble           → item_forecasts
(chained)     Backtest              → accuracy tracking           → prediction_accuracy
Weekly Sun    Event Correlation     → market-event price impacts  → analysis tables
```

---

## Scheduled Workflows

| Workflow | Schedule | Purpose |
|----------|----------|---------|
| `aggregator-update` | Daily 23:00 UTC | Multi-source collection + Parquet archive commit |
| `supply-scraper` | Daily 22:00 UTC | Listing supply-depth per item |
| `price-forecast` | Chained off aggregator | LightGBM predictions (predict-only Tue–Sun, full retrain Mon) |
| `backtest-accuracy` | Chained + M–Sat 08:00 UTC | Daily forecast accuracy evaluation |
| `event-correlation-analysis` | Weekly Sun 04:00 UTC | Market-event price impacts |
| `discover-new-items` | Ad hoc | Scan for newly tradable items |
| `reddit-sentiment` | Every 6h | Community sentiment analysis |

---

## Project Structure

```
cs2-oracle/
├── backend/              # FastAPI server, routes, collectors, models, tests
│   ├── api/              # Route handlers
│   ├── collectors/       # Market data collectors (7 sources)
│   ├── models/           # SQLAlchemy ORM models
│   ├── analytics/        # ML forecasting, backtesting, event analysis
│   ├── scripts/          # Task runner + utilities
│   └── tests/            # Pytest suite
├── frontend/             # Next.js app router + components
│   ├── app/              # Pages (app router)
│   ├── components/       # React component library
│   └── lib/              # API client, utilities
├── docs/                 # Architecture docs, changelog, research
├── price-archive/        # Parquet price data (13 years)
└── .github/workflows/    # 7 CI/CD automation workflows
```

---

## Security

- Never commit `.env` files — use `.env.example` as template
- Replace default `SECRET_KEY` in production deployments
- Enable GitHub Secret Scanning for the repository
- Pre-commit hooks via `.pre-commit-config.yaml` enforce secrets check

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Install pre-commit hooks: `pre-commit install`
4. Make changes and verify with tests: `pytest`
5. Commit using conventional commit messages
6. Open a pull request
