# CS2 Market Analyzer

Full-stack analytics platform for Counter-Strike 2 skin market data. Daily pipeline collects multi-source prices from 7 markets, archives to Parquet, serves via FastAPI to a Next.js dashboard, and forecasts via LightGBM quantile ensembles.

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
.github/workflows/  — 7 CI/CD workflows
```

## Commands

### Backend
- `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt` — setup (from backend/)
- `uvicorn main:app --host 0.0.0.0 --port 8000` — start server (from backend/)
- `pytest` — run all tests (from backend/)
- `pytest tests/test_file.py -k "test_name"` — run specific test
- `python3 -m py_compile main.py` — syntax check
- `alembic upgrade head` — run migrations

### Frontend
- `npm install` — install dependencies (from frontend/)
- `npm run dev` — dev server
- `npm run build` — type-check + production build
- `npm run lint` — ESLint

## Important Gotchas

- **Next.js 16 has breaking changes.** Read `node_modules/next/dist/docs/` before writing any new code. Do not assume APIs match training data.
- **The `daily_analysis` table was dropped** (migration 0015). Do not reference it in code or queries. Functionality superseded by `item_forecasts` + Parquet-based analysis.
- **API client lives in `frontend/lib/api.ts`.** When adding routes, update both the backend router and this client.
- **Use CSS custom properties from `frontend/app/globals.css`** — never inline hex values. OKLCH color space.
- **Use Tabular-nums on all numeric content.** Inter for UI, JetBrains Mono for data.
- **Regime-switching:** `forecaster.py` trains separate models per market regime (bear/range/bull). Regime models stored in `self.regime_models[(regime, horizon, q)]`. Use `--compare-regime` in `forecast_prices.py` to run A/B comparison. New regime models add ~23 min to retrain.

## Design Tokens (OKLCH)

- Bg: `oklch(18% 0.004 30)` / `oklch(22% 0.005 30)` / `oklch(15% 0.003 30)`
- Text: `oklch(93% 0 0)` / `oklch(70% 0.004 30)` / `oklch(52% 0.004 30)` / `oklch(38% 0.003 30)`
- Accent: `oklch(62% 0.14 55)` — warm amber
- Data up: `oklch(62% 0.14 155)` / down: `oklch(62% 0.12 25)`
- Radii: 2px (xs), 4px (sm), 6px (md)

## Workflow Rules

1. Backend changes: run `pytest` and `python3 -m py_compile` before marking done.
2. Frontend changes: run `npm run lint` and `npm run build` before marking done.
3. When adding API routes, also update `frontend/lib/api.ts` with the typed fetch function.
4. Keep `frontend/AGENTS.md` in sync if design tokens or API surface changes.
5. The `@review` subagent can check changes automatically — invoke it after significant work. The chain plugin (`.opencode/plugins/chain.js`) auto-triggers review → test → security after build edits; a 5-min cooldown prevents loops.
6. Use `@data` for DuckDB/Parquet queries on the `price-archive/`. It knows the Parquet schemas, common DuckDB patterns, and schema gotchas (VARCHAR pricing columns, missing source in older files).
7. When adding agents, update both `opencode.json` task permissions and this file.
