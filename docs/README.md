# docs/

Refreshed 2026-08-05 against the code. Where a doc and the code disagree, the code wins —
report it rather than working around it.

## Architecture (`architecture/`)

- `model.md` — the forecaster as it stands: 4 q50 LightGBM models + 4 directional
  classifiers, the split-conformal band, sequential training, age-based retrain
- `model-optimization.md` — size/speed levers, split into already-applied, still-available,
  and 🛑 do-not
- `pipeline.md` — the aggregator and the workflows chained off it; what collects and what
  no longer does
- `data.md` — Parquet archive layout, the `ops/` mirror layer, Supabase serving tables

> **No production directional-accuracy figure is currently quotable.**
> `MIN_FORECAST_DATES = 20` (`backend/backtest/scoring.py`) and live cohorts span 1–2
> distinct forecast dates, so every horizon reports NO HEADLINE. This is a calendar
> problem, not a code problem. Offline CV DA and production DA are not comparable until
> the served series accumulates ~20 dates.

## Reference (`references/`)

- `steam-api.md` — Steam Market endpoints and response formats, empirically tested.
  The rate-limit envelope applies to **residential IPs only** — hosted CI runners are
  429'd on the first request.
- `data-sources.md` — per-source status, freshness, known issues
- `catalog-build.md` — Steam catalog scrape: rate-limiting strategy, gap repair
- `backfill.md` — **Dead capability.** CSMarketAPI multi-market backfill; the free-key
  quota never resets and the local DB is empty. Kept for the key-rotation and
  priority-queue design only.

## Research (`research/`)

- `2026-08-07-cs2-forecasting-research.md` — **Start here for anything accuracy-related.**
  1,706-line external design review. Its C1–C5 correction block overturns four of its own
  first-pass claims and three of those touch live code: a BUFF **bid** may be voting into the
  consensus price as an ask since 2026-07-11, the 1.1607 Steam fee constant is synthetic, and
  `walkforward_backtest.py --purge` is default OFF so the published backtest number is
  unpurged. Also carries the one measured positive — expensive tiers lead cheap tiers by a
  day, z = 9.1. See `changelog/2026-08-07-cs2-forecasting-research-review.md`.
- `2026-08-07-next-steps.md` — the tracked, **gated** action list from that review. Step 1 is
  one query and gates everything else; steps 1–6 need no retrain and are ~2 weeks. Everything
  on it is NOT STARTED.
- `accuracy-opportunities.md` — closed 2026-07-31, **reopened 2026-08-07** by the review
  above, which relocates the binding constraint from input data to measurement. The stop
  banner is intact and the tables are still a record of what was tried, not a backlog. Read
  both before proposing accuracy work.
- `2026-07-19-feature-contribution-by-horizon.md` — ablation behind the
  `HORIZON_EXCLUDED_GROUPS` config in `models/forecaster.py`
- `2026-07-21-training-time-optimization.md` — training-time levers
- `volume-data.md` — volume evaluation: free source in the archive, zero predictive lift
- `competitor-analysis.md` — landscape and differentiators
- `2026-07-27-direction-label-sweep-raw.txt` — raw sweep output

## Design docs and plans (`superpowers/`)

`specs/` holds designs (8), `plans/` the execution checklists (5). Each shipped change is
also recorded in `changelog/`, which is the durable record. Load-bearing ones:

- `specs/2026-07-25-monthly-parquet-partitioning-design.md` — the live partitioning scheme
  in `scripts/append_to_parquet.py`
- `specs/2026-08-01-deterministic-backtest-design.md` — shared-estimator backtest
- `specs/2026-08-03-served-forecast-surface-design.md` — confidence-gate removal, $1 floor
- `specs/2026-08-04-minimal-model-design.md` — the 40→8 model collapse
- `specs/2026-08-05-cv-cohort-parity-design.md` — CV/production cohort mismatch. Its
  residual-gap table rests on 1–2 market days; read it with the NO HEADLINE caveat above.

## Changelog (`changelog/`)

Append-only dated decision records: bug fixes, features, audits, and refuted experiments.
57 entries, 2026-07-08 to 2026-08-06. Entries are never edited to match later reality —
several describe code that has since been deleted, which is the point. Per `AGENTS.md`
rule 4, non-trivial decisions get a new dated note here.

## Other

- `code-review-2026-07-21.md` — **Live punch list**, findings re-verified 2026-08-05.
  Most are still open, and the security findings cluster (SQL f-strings, default secret
  key, session token in a redirect URL). Separates LIVE from DORMANT.
- `operations.md` — runbook: workflow schedules, required secrets, load-bearing steps,
  troubleshooting
- `design.md` — visual design system: OKLCH palette, typography, spacing, components
- `product.md` — positioning, users, brand personality, design principles

## Removed 2026-08-05

`historical/` (5 files) and `retrain-optimization-analysis.md` were deleted — the first
documented only resolved issues, the second optimized a 36-model quantile grid that no
longer exists. Both are recoverable from git history if needed.
