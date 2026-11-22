# The API is retired (Phase 4, PR 5)

2026-11-22. Part of `specs/2026-11-09-delete-postgres-design.md` (Q1, decided 2026-11-09: retire
everything), plan step "PR 5". Stacked on PR 4 (`2026-11-22-forecaster-holds-no-session.md`).

**No API until the frontend rebuild.** The API was never deployed and nothing called it; the
frontend it fed was deleted on 2026-08-10. Its routes read `PriceHistory` (frozen since
2026-07-11), `users` and `events`, which kept Postgres alive for no consumer. The product until
the rebuild is the Parquet forecast record (`ops/item_forecasts/`) and the backtest's scores
(`ops/prediction_accuracy.parquet`) in the data repo.

## Deleted

- `main.py` and all of `api/`: routes (`accuracy`, `auth`, `events`, `items`, `market`,
  `monitoring`, `opportunities`), `schemas.py`, `cache.py`, `serving_policy.py`,
  `volatility_tags.py`, and the 15 test files that tested only them. Mixed test files keep their
  non-API tests (clean-anchor disclosure, phase-collapsed and phantom universes, direction
  disposition, the `confidence` instrument).
- `scripts/sync_events_from_news.py`, `scripts/event_correlation_analysis.py` and their tests, and
  the `event_correlation` task in `run_task.py`. They fed nothing served (spec Q2).
  `NO_OP_STATUSES` is now empty; the mechanism stays.
- The `serving-policy` Claude rule, scoped to `backend/api/**`.

## Kept or moved

- `MIN_SERVED_PRICE_USD = 1.0` moved to `backtest/scoring.py`, beside `HEADLINE_MIN_TIER`, which it
  must equal (`tests/test_served_price_floor.py`). `outside_baseline.py`, `replay_serving.py`,
  three archived harnesses and a training-coverage test import it from there. The convention
  note moved into the `backtest-scoring` rule.
- Invariant 5 (shadow predictions never reach `item_forecasts` or public schemas): with no public
  schemas left, its two tests check the `item_forecasts` partition schema instead of
  `api/schemas.py`.
- The ORM models, including `User`, stay as the test schema (spec §2). `config.py`'s
  `secret_key` and the `fastapi` / `uvicorn` dependencies are left for the PR 6 cleanup.
- The served-band disclosure columns `predict` writes (`anchor_clean`, `anchor_wedge_pct`,
  `exceed_p`) are unchanged: they are the forecast record, not the API.
