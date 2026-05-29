# CS2 Market Analyzer

CS2 Market Analyzer is a Counter-Strike 2 market intelligence project for collecting, processing, and visualizing item price data. The backend handles collection, validation, migration, pruning, and trend analysis; the frontend provides a modern dashboard for exploring the data.

## Contents

- [Overview](#overview)
- [Features](#features)
- [Stack](#stack)
- [Repository structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Environment variables](#environment-variables)
- [Setup](#setup)
- [Common commands](#common-commands)
- [Docs](#docs)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)

## Overview

The project is organized as a full-stack analytics platform:

- **Backend**: Python data pipeline, collectors, validation, migrations, and scheduled maintenance tasks
- **Frontend**: Next.js dashboard that consumes the backend API configured through `NEXT_PUBLIC_API_URL`
- **Automation**: GitHub Actions workflows for recurring collection and database maintenance

## Features

- Market data collection from CS2-related sources
- Data validation and cleanup before persistence
- Trend and opportunity analysis for long-term item tracking
- Pruning and downsampling jobs for historical data management
- Interactive frontend charts and market views
- Scheduled workflow automation for recurring jobs

## Stack

| Layer | Tech |
| --- | --- |
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS 4, Recharts, Framer Motion |
| Backend | Python, SQLAlchemy, Alembic, Pydantic Settings, Requests, HTTPX |
| Data | PostgreSQL / Supabase compatible storage |
| Automation | GitHub Actions |

## Repository structure

| Path | Purpose |
| --- | --- |
| `backend/` | Collectors, analytics, migration config, scripts, and tests |
| `frontend/` | Next.js application and client-side API layer |
| `.github/workflows/` | Scheduled collection and maintenance workflows |
| `PRODUCT.md` | Product positioning and audience notes |
| `DESIGN.md` | Visual and UX direction for the interface |
| `WORKFLOW_MONITORING.md` | Operational guide for scheduled jobs |

## Prerequisites

- Python 3.9+
- Node.js 18+
- npm
- PostgreSQL or Supabase

## Environment variables

Create local environment files before running anything:

### `backend/.env`

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | Database connection string |
| `ENVIRONMENT` | `development` or `production` |
| `DEBUG` | Enables local debug behavior |
| `STEAM_API_KEY` | Optional Steam integration key |
| `CS2SH_API_KEY` | Optional secondary market API key |
| `FRONTEND_URL` | Frontend origin used by backend components |
| `SECRET_KEY` | Replace the default value for production |

### `frontend/.env.local`

| Variable | Purpose |
| --- | --- |
| `NEXT_PUBLIC_API_URL` | Backend API base URL, defaults to `http://localhost:8000` |

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/RayanR000/cs2-market-analyzer.git
cd cs2-market-analyzer
```

### 2. Configure the backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env
```

Update `backend/.env` with your database URL and any optional keys.

### 3. Run database migrations

```bash
cd backend
source venv/bin/activate
python scripts/run_task.py migrate
```

### 4. Configure the frontend

```bash
cd frontend
npm install
```

Create `frontend/.env.local` and set `NEXT_PUBLIC_API_URL` if your backend runs somewhere other than `http://localhost:8000`.

### 5. Start development

```bash
cd frontend
npm run dev
```

The frontend runs on `http://localhost:3000`.

## Common commands

### Backend tasks

```bash
cd backend
source venv/bin/activate
python scripts/run_task.py aggregate
python scripts/run_task.py priority
python scripts/run_task.py prune
python scripts/run_task.py trends
python scripts/run_task.py long_term_trends
python scripts/run_task.py migrate
```

### Frontend tasks

```bash
cd frontend
npm run dev
npm run lint
npm run build
npm run start
```

### Tests

```bash
cd backend
source venv/bin/activate
pytest
```

## Docs

- `PRODUCT.md` for product goals and audience
- `DESIGN.md` for visual direction and UI principles
- `WORKFLOW_MONITORING.md` for scheduled job monitoring
- `backend/data/HISTORICAL_DATA_SOURCES.md` for data provenance notes
- `frontend/README.md` for default Next.js project guidance

## Testing

Recommended checks before merging:

- `cd backend && pytest`
- `cd frontend && npm run lint`
- `cd frontend && npm run build`

## Deployment

- **Frontend**: deploy as a standard Next.js application
- **Backend jobs**: run through GitHub Actions or your preferred scheduler
- **Database**: point `DATABASE_URL` at your production Postgres or Supabase instance

## Contributing

1. Keep changes focused and documented.
2. Update migrations when the schema changes.
3. Run the relevant backend and frontend checks before opening a PR.
4. Add or update docs when behavior changes.

## Security

- Never commit `.env` files or secrets.
- Replace the default backend `SECRET_KEY` in production.
- Restrict database credentials to the minimum required permissions.
- Review workflow secrets before enabling scheduled jobs in a production repository.

## License

No license file is currently included in this repository.
