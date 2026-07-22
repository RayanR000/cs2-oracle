# CS2 Oracle

Daily pipeline collects multi-source prices from 7 markets, archives to Parquet, serves via FastAPI to a Next.js dashboard, and forecasts via LightGBM quantile ensembles.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS 4, Recharts, Framer Motion |
| Backend | Python 3.11, FastAPI, SQLAlchemy 2, Alembic, Pydantic Settings |
| Data | PostgreSQL / Supabase, DuckDB, Parquet, Pandas, NumPy, SciPy |
| ML | LightGBM, Optuna |
| Automation | GitHub Actions (7 workflows) |

## Project Structure

```
backend/            — FastAPI server, routes, collectors, models, tests
frontend/           — Next.js app router, components, API client
docs/               — Architecture, changelog, references, research
price-archive/      — Parquet price data (13 years)
price-archive/ops/  — Parquet operational data (events, forecasts, accuracy, supply, etc.)
.github/workflows/  — 7 CI/CD workflows
```

## Commands

### Backend (from `backend/`)
- `uvicorn main:app --host 0.0.0.0 --port 8000` — start server
- `pytest` — run all tests
- `pytest tests/test_file.py -k "test_name"` — run specific test
- `python3 -m py_compile main.py` — syntax check
- `alembic upgrade head` — run migrations

## Gotchas

- **Training data from Parquet, not DB.** `fetch_price_history(backfilled_only=True)` reads `price-archive/*.parquet` via DuckDB. DB only used for `is_backfilled` flag + events metadata.
- **Operational tables migrated to `price-archive/ops/*.parquet`.** API routes read Parquet first with DB fallback. See `backend/db/parquet.py`.
- **API client at `frontend/lib/api.ts`.** Update both backend router and this client when adding routes.
- **Social sentiment features are non-functional.** VADER scores CS2 jargon as neutral — features don't rank in top 20 by gain.

## Workflow Rules

1. Run `pytest` + `python3 -m py_compile` for backend changes.
2. Run `npm run lint` + `npm run build` for frontend changes.
3. When adding API routes, also update `frontend/lib/api.ts`.
4. Keep `frontend/AGENTS.md` in sync if design tokens or API surface changes.
5. For frontend design, see `frontend/AGENTS.md` (OKLCH tokens, typography, styling rules).
6. Use subagents: `@review` after significant work, `@data` for Parquet queries, `@explore` for codebase search, `@document` for changelog/architecture.
7. When adding agents, update `opencode.json` task permissions and this file.
