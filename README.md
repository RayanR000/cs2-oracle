<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/CS2%20Market%20Analyzer-121212?style=for-the-badge&logo=counter-strike&logoColor=white">
    <img alt="CS2 Market Analyzer" src="https://img.shields.io/badge/CS2%20Market%20Analyzer-FFFFFF?style=for-the-badge&logo=counter-strike&logoColor=black" width="320">
  </picture>
</p>

<p align="center">
  <em>Counter-Strike 2 market intelligence — collect, analyze, and visualize item price data.</em>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#features">Features</a> •
  <a href="#stack">Stack</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#usage">Usage</a> •
  <a href="#deployment">Deployment</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/node-18%2B-339933?style=flat-square&logo=node.js&logoColor=white" alt="Node">
  <img src="https://img.shields.io/badge/next.js-16-000000?style=flat-square&logo=next.js&logoColor=white" alt="Next.js">
  <img src="https://img.shields.io/badge/fastapi-0.115%2B-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/postgresql-4169E1?style=flat-square&logo=postgresql&logoColor=white" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/license-MIT-yellow?style=flat-square" alt="License">
</p>

---

## Overview

CS2 Market Analyzer is a full-stack analytics platform that collects, validates, and visualizes Counter-Strike 2 skin market data. A daily pipeline pulls multi-source prices from CSGOTrader (Steam, Skinport, Buff163, CSFloat, CSMoney, Youpin), writes them to a Parquet archive, and serves them through a FastAPI REST API to the Next.js frontend. ML price forecasts (LightGBM quantile ensembles) replace traditional trend analysis for all signal generation.

```
┌──────────────────┐     ┌──────────────┐     ┌──────────────┐
│  CSGOTrader API   │────▶│    API / DB   │────▶│   Dashboard  │
│  7 market sources │     │  (FastAPI)   │     │  (Next.js)   │
└──────────────────┘     └──────────────┘     └──────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
   Parquet Archive        Supabase PG              Recharts
   (data-archive branch)  (daily closes)           + Framer
```

## Features

- **Multi-Source Collection** — daily aggregator fetches prices from 7 market sources (Steam, Skinport, Buff163, CSFloat, CSMoney, CSGOTrader, Youpin)
- **Parquet Price Archive** — complete historical price data from 2013 onward, queryable via DuckDB
- **ML Price Forecasts** — LightGBM quantile regression (q10/q50/q90) across 3, 7, 14, and 30-day horizons, trained as a 6-member diversified ensemble with walk-forward validation and Optuna hyperparameter tuning over a 1460-day window
- **Forecast Accuracy Tracking** — automated daily backtesting with MAE, MAPE, directional accuracy, and concept drift alerts
- **Model Explainability** — per-item feature importance exposes which signals drive each forecast
- **Event Impact Analysis** — quantifies how market events (operations, cases, pro matches) move individual item prices
- **Player Count & Supply Signals** — Steam player counts and listing supply depth are collected (supply depth used in forecasts; player counts separately archived)
- **Technical Signals** — SMA, Bollinger Bands, RSI, MACD, support/resistance computed per item
- **Market Opportunities** — undervalued, overheated, and momentum signals derived from ML forecasts
- **Quality Variant Grouping** — items grouped by base name with all wear levels and special variants
- **Interactive Dashboard** — responsive charts, grouped market views, item detail with multi-source pricing
- **Steam Authentication** — OpenID login with session management
- **Scheduled Automation** — collection, forecast, backtest, event correlation, player-count, and supply scraping via GitHub Actions

## Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Next.js 16, React 19, TypeScript, Tailwind CSS 4, Recharts, Framer Motion |
| **Backend** | Python 3.11, FastAPI, SQLAlchemy, Alembic, Pydantic Settings |
| **Data** | PostgreSQL / Supabase, DuckDB, Parquet, Pandas, NumPy, SciPy |
| **ML** | LightGBM, Optuna (hyperparameter tuning, 15–30 trials) |
| **Storage** | Git data-archive branch for Parquet price archive |
| **Automation** | GitHub Actions (7 workflows) |

## Repository Structure

```
├── backend/
│   ├── analytics/        # (deprecated — trend analysis removed)
│   ├── api/              # FastAPI route handlers + schemas + cache
│   │   ├── routes/       # items, market, opportunities, events, auth, portfolio, accuracy
│   │   ├── cache.py      # In-process TTL cache
│   │   └── schemas.py    # Pydantic response models
│   ├── collectors/       # CSGOTrader aggregator (7 sources), pipeline orchestration
│   ├── models/           # SQLAlchemy ORM models + LightGBM forecaster (models retrained on schedule, not committed)
│   ├── migrations/       # Alembic migration scripts
│   ├── scripts/          # Task runners, backtest, forecast, Parquet export
   │   ├── tests/            # pytest suite (~170 tests)
│   ├── main.py           # FastAPI app entry point
│   ├── config.py         # Pydantic settings
│   └── database.py       # All SQLAlchemy models + session management
├── frontend/
│   ├── app/              # Next.js App Router pages (home, market, items/[id], portfolio, accuracy)
│   ├── components/       # Header, Search, StatCard, ItemCard, etc.
│   └── lib/              # API client, ThemeContext, UserContext
├── price-archive/        # Parquet price data (13 years of history)
├── .github/workflows/    # 7 scheduled workflows
├── docs/                 # Architecture, research, changelogs, and reference docs
│   ├── architecture/     # Data architecture, model architecture, pipeline
│   ├── references/       # Steam API, data sources, catalog build, backfill
│   ├── research/         # Feature engineering, accuracy opportunities
│   ├── historical/       # Backend review, session notes, migration plans
│   └── changelog/       # Dated execution logs and bug fixes
```

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 14+ (or Supabase account)

### 1. Clone

```bash
git clone https://github.com/RayanR000/cs2-market-analyzer.git
cd cs2-market-analyzer
```

### 2. Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit with your database URL
```

Configure `backend/.env`:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `ENVIRONMENT` | `development` or `production` |
| `DEBUG` | Enable debug logging |
| `STEAM_API_KEY` | (Optional) Steam Web API key |
| `CSMARKETAPI_KEY_{1-6}` | (Optional) CSMarketAPI key for backfill |
| `CSMARKETAPI_ACCOUNT_{1-6}` | Account name for key tracking |
| `STEAM_SESSION_ID` | (Optional) Steam session cookie for price history |
| `STEAM_LOGIN_SECURE` | (Optional) Steam login cookie |
| `FRONTEND_URL` | Frontend origin for CORS |
| `SECRET_KEY` | Session signing secret (rotate in production) |

### 3. Database Migrations

```bash
cd backend
source venv/bin/activate
python scripts/run_task.py migrate
```

### 4. Frontend Setup

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend API base URL |

### 5. Start Development

```bash
# Terminal 1 — Backend API
cd backend && source venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend
cd frontend && npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Usage

### Backend Tasks

```bash
cd backend
source venv/bin/activate
python scripts/run_task.py <task>
```

| Task | Description |
|------|-------------|
| `aggregate` | Full aggregator collection (all items, all sources) |
| `priority` | Top 2000 items collection |
| `trends` | Deprecated (no-op — ML forecasts handle signal generation) |
| `long_term_trends` | Deprecated (no-op) |
| `migrate` | Run pending Alembic migrations |
| `backtest` | Run forecast accuracy backtest (all types) |
| `backtest_historical` | Run historical walk-forward backtest |

### Frontend Commands

```bash
cd frontend
npm run dev     # Development server
npm run build   # Production build
npm run start   # Start production server
npm run lint    # Run ESLint
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/items/` | List items (paginated, backfilled only) |
| GET | `/items/count` | Total backfilled item count |
| GET | `/items/search?q=` | Search items by name |
| GET | `/items/trending` | Trending items from ML forecasts |
| GET | `/items/{id}` | Item details |
| GET | `/items/{id}/price-history` | Price history (SMA included) |
| GET | `/items/{id}/trends` | Technical signals (RSI, Bollinger, MACD, support/resistance) |
| GET | `/items/{id}/prediction` | ML price forecast (7d or 30d) |
| GET | `/items/{id}/variants` | Quality variants grouped by base name |
| GET | `/items/{id}/prices` | Multi-source price data |
| GET | `/items/{id}/events` | Market events affecting item |
| GET | `/items/{id}/event-impacts` | Quantified price impact of market events |
| GET | `/items/{id}/feature-importance` | Per-item forecast feature importance |
| GET | `/market/summary` | Grouped market view (paginated, cached) |
| GET | `/opportunities/` | All market opportunities |
| GET | `/opportunities/undervalued` | Undervalued items |
| GET | `/opportunities/overheated` | Overheated items |
| GET | `/opportunities/momentum` | Momentum items |
| GET | `/events/` | Market events |
| GET | `/events/recent` | Recent events |
| GET | `/accuracy/` | Forecast accuracy records |
| GET | `/accuracy/latest` | Latest accuracy per type |
| GET | `/accuracy/summary` | Aggregated accuracy summary |
| GET | `/accuracy/outcomes` | Per-forecast correctness |
| GET | `/accuracy/outcomes/stats` | Overall accuracy statistics |
| GET | `/auth/me` | Current user |
| GET | `/auth/steam/login` | Steam OpenID login |
| GET | `/auth/callback` | Steam auth callback |
| POST | `/auth/logout` | Logout |
| GET | `/portfolio/inventory` | User portfolio |
| GET | `/ab-test/regime` | Regime vs global-only model A/B comparison |
| GET | `/ab-test/ensemble` | Ensemble size A/B comparison (3 vs 6) |

### Testing

```bash
cd backend && source venv/bin/activate && pytest
```

> ~170 tests across collection, forecasting, backtesting, and API layers.

### Pre-commit Checks

```bash
pip install pre-commit detect-secrets
pre-commit install
detect-secrets scan > .secrets.baseline  # optional
```

## Deployment

### Frontend

Deploy as a standard Next.js application (Vercel recommended):

```bash
cd frontend
npm run build
npm run start   # or deploy via Vercel CLI / GitHub import
```

### Backend

The backend runs on-demand via scheduled GitHub Actions. For a persistent API server:

```bash
cd backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Scheduled Workflows

Seven GitHub Actions workflows handle recurring operations:

| Workflow | Schedule | Purpose |
|----------|----------|---------|
| `aggregator-update` | Daily 23:00 UTC | Multi-source price collection + Parquet archive commit |
| `supply-scraper` | Daily 22:00 UTC | Listing supply-depth collection per item |
| `player-count-hourly` | Every 2h | Steam player-count tracking as a market-regime signal |
| `price-forecast` | Chained off aggregator | LightGBM predictions (predict-only Tue–Sun, full retrain Mon) |
| `backtest-accuracy` | Chained off forecast + cron 08:00 UTC (Mon–Sat) | Daily forecast accuracy evaluation |
| `event-correlation-analysis` | Weekly Sun 04:00 UTC | Quantifies market-event price impacts |
| `discover-new-items` | Manual dispatch only | Steam item discovery (dead — catalog curated via backfill) |

### Data Flow

```
22:00  Supply Scraper → listing supply depth → Supabase
23:00  Aggregator → 7 source prices → CSV → Parquet (data-archive branch)
                                        → Supabase (aggregator_sync only)
Every 2h  Player Count → Steam active players → Supabase + Parquet
Chained  Forecast → LightGBM ensemble → item_forecasts table
Chained  Backtest → accuracy tracking → prediction_accuracy + forecast_outcomes
Weekly   Event Correlation → event-impact analysis
```

### Database

Point `DATABASE_URL` to your production PostgreSQL or Supabase instance. Run migrations:

```bash
python scripts/run_task.py migrate
```

## Security

- Never commit `.env` files — the repository includes `.env.example` as a template
- Replace the default `SECRET_KEY` in production
- Enable [GitHub Secret Scanning](https://docs.github.com/en/code-security/secret-scanning) for your repository
- Restrict database credentials to the minimum required permissions

## Contributing

1. Keep changes focused and well-documented.
2. Update Alembic migrations when the schema changes.
3. Run `pytest` and `npm run lint` before opening a PR.
4. Update docs when behavior changes.

## License

MIT — see [LICENSE](LICENSE) for details.
