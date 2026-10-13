# Paused workflows retired; lint covers the whole repo

**Date:** 2026-10-13

## What changed

- **Deleted** `event-correlation-analysis.yml`, `ab-harness-batch.yml` and `discover-new-items.yml`,
  all `disabled_manually` since the 2026-08-23 Actions-minutes pause, plus the one-off
  `ops-partition-backfill.yml` (applied 2026-10-12, run 38295392813). `discover-new-items` could not
  run anyway: its script imported a `collectors.real_data_collector` that no longer existed. The
  scripts they called remain; git history keeps the YAML.
- **`model-diagnostics.yml` is dispatch-only** and enabled again. Six tests read its inputs and two
  rules cite it, so it stays. A dispatch with every input at its default is the control arm the
  Sunday cron used to be.
- **Lint runs `ruff check .` and `ruff format --check backend`** (migrations excluded) instead of
  linting only the files a push changed. The three findings outside that gate were fixed and 56
  `backend/` files reformatted; `ruff format` preserves the AST, and the fast gate (2,867 passed),
  mypy and lint are unchanged.
- `docs/architecture/pipeline.md`, `docs/operations.md`, the README and the pipeline-auditor agent
  list the nine remaining workflows. The README no longer links the private `cs2-oracle-data` repo.
