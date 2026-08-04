# CS2 Oracle

Daily pipeline collects multi-source prices from 7 markets, archives to Parquet, serves via FastAPI to a Next.js dashboard, and forecasts via LightGBM quantile ensembles.

## Gotchas

- **API client at `frontend/lib/api.ts`.** Update both backend router and this client when adding routes.
- **`--predict-only` does not retrain.** It used to: `check_concept_drift` ran against a hardcoded 60% floor above the model's measured 46.7–50.8% DA, so drift fired every run and cost a measured 465s of the 835s daily step. Drift is now reported only. Set `ALLOW_DRIFT_RETRAIN=1`, or dispatch `price-forecast.yml` with `mode=full`, to retrain.

## Workflow Rules

1. Run `pytest` + `python3 -m py_compile` for backend changes.
2. Run `npm run lint` + `npm run build` for frontend changes.
3. When adding API routes, also update `frontend/lib/api.ts`.
4. Keep `frontend/AGENTS.md` in sync if design tokens or API surface changes.
5. For frontend design, see `frontend/AGENTS.md` (OKLCH tokens, typography, styling rules).
6. Use subagents: `@review` after significant work, `@data` for Parquet queries, `@explore` for codebase search, `@document` for changelog/architecture.
7. When adding agents, update `opencode.json` task permissions and this file.
