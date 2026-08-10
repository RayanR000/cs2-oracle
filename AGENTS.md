# CS2 Oracle

Daily pipeline pulls 7 markets' prices from the csgotrader.app dumps, archives them to
Parquet, serves them via FastAPI, and forecasts with LightGBM — one q50 model per horizon,
with the served band calibrated by split conformal.

**There is no frontend.** It was deleted on 2026-08-10 to be rebuilt from scratch; see
`docs/changelog/2026-08-10-frontend-removed.md`. The API is the product surface for now.

- `backend/` — FastAPI (`main.py`), routers in `api/routes/`, ML in `models/forecaster.py`,
  batch jobs in `scripts/`. See `backend/AGENTS.md`.
- `price-archive/` — gitignored **plain local directory**, not a symlink and not a
  checkout. Editing it changes nothing in production: the durable archive is the separate
  `RayanR000/cs2-oracle-data` repo, which only CI writes (orphan commit + force-push in
  `aggregator-update.yml`). The local copy also runs *behind* it. Nothing written there is
  committed by this repo.
- `docs/` — `architecture/` (4), dated decision records in `changelog/` (119), `references/` (5),
  `research/` (14, incl. the live action list), `superpowers/{specs,plans}` (28), and the loose
  `design.md`, `product.md`, `operations.md`. Indexed in `docs/README.md`.
- `.claude/rules/` — backend subsystem detail, scoped by path so it loads only when you touch
  the matching files. Indexed in `backend/AGENTS.md`.
- `.github/workflows/` — daily chain: Aggregator (23:00 UTC) → Price Forecast → Backtest
  Accuracy, each chained off the previous run's success.

## Commands

Backend, from `backend/`, through the venv (`venv/bin/python`, Python 3.13 locally / 3.11 in CI):

- `venv/bin/python -m pytest tests/test_<name>.py -q` — targeted run.
- `venv/bin/python -m pytest tests/ -q` — full suite. Collection is ~1.4s. It used to take
  ~45s and to hit **production** on the way: `tests/test_price_history.py` had no test
  functions at all, just module-level code that opened a prod session and made five live
  Steam calls with `time.sleep(10)` between them. Deleted 2026-08-10, along with
  `scripts/test_social_signal.py`, which was why a bare `pytest -q` used to abort.
- `venv/bin/uvicorn main:app --port 8000` — the API.

## Gotchas

- **`backend/.env` points at production.** It sets the prod Supabase pooler
  `DATABASE_URL` and `ENVIRONMENT=production`, and `config.py` reads `.env` relative to
  the **current working directory** — so a script run from `backend/` writes to prod,
  while the same command from the repo root picks up the root `.env` (no `DATABASE_URL`,
  `ENVIRONMENT=development`). `database.py` builds the engine at import, so setting
  `os.environ["DATABASE_URL"]` inside a script is already too late to redirect it.
- **`--predict-only` does not retrain.** It used to: `check_concept_drift` ran against a
  hardcoded 60% floor above the model's measured 46.7–50.8% DA, so drift fired every run
  and cost a measured 465s of the 835s daily step. Drift is now reported only. Set
  `ALLOW_DRIFT_RETRAIN=1`, or dispatch `price-forecast.yml` with `mode=full`, to retrain.
- **A task that returns no count fields defeats the zero-row guard.** `scripts/run_task.py`
  fails a run when a task reports row counts and every one is zero — that is the only
  thing standing between a dead collector and a green badge. Two collectors hid behind an
  earlier guard that keyed on a single field. New tasks must return their counts.

## Workflow Rules

1. Backend changes: run the relevant `pytest` files.
2. Non-trivial decisions get a dated note in `docs/changelog/`.
3. `docs/design.md` describes the deleted frontend. It is kept as rebuild input only —
   do not treat it as a spec for anything that currently runs.
