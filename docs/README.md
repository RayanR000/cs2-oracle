# docs/

## Architecture (`architecture/`)

- `data.md` — Parquet archive, Supabase serving layer, storage breakdown, migration history
- `model.md` — LightGBM forecaster: features, ensembles, regime-switching, accuracy, parameters
- `model-optimization.md` — Size/speed levers and the measured effect of each
- `pipeline.md` — CSGOTrader aggregator: multi-source collection, Parquet storage, coverage

> **Accuracy figures in `model.md` and `model-optimization.md` are not current.** See
> `changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md` — no directional-accuracy
> number is reportable until the served series spans 20 distinct forecast dates.

## Reference (`references/`)

- `steam-api.md` — Steam Market API endpoints, rate limits, response format (empirically tested)
- `backfill.md` — CSMarketAPI multi-market backfill: key rotation, priority queue, execution results
- `data-sources.md` — Source quality, freshness, known issues, volume data analysis
- `catalog-build.md` — Phase 1 Steam catalog scrape: rate-limiting strategy, gap repair

## Research (`research/`)

- `accuracy-opportunities.md` — **Closed 2026-07-31.** Carries a stop banner; the remaining
  items were abandoned as unmeasurable, not deferred. Read before proposing accuracy work.
- `2026-07-19-feature-contribution-by-horizon.md` — Ablation study behind the live
  `HORIZON_EXCLUDED_GROUPS` config in `models/forecaster.py`
- `2026-07-21-training-time-optimization.md` — Training-time levers
- `volume-data.md` — Volume data evaluation: free source in archive, zero predictive lift verified
- `competitor-analysis.md` — Landscape: CSMarketCap, SteamAnalyst, TradeUp Academy, differentiators

## Design docs and plans (`superpowers/`)

`specs/` holds designs, `plans/` the execution checklists. Only those still load-bearing are
kept — each shipped change is recorded in `changelog/`, which is the durable record.

- `specs/2026-08-03-served-forecast-surface-design.md` + `plans/…` — confidence-gate removal, $1 floor
- `specs/2026-08-01-deterministic-backtest-design.md` + `plans/…` — shared-estimator backtest
- `specs/2026-07-25-monthly-parquet-partitioning-design.md` — the live partitioning scheme in
  `scripts/append_to_parquet.py`
- `specs/2026-07-25-quality-spread-features-design.md` — referenced by `accuracy-opportunities.md`

## Historical (`historical/`)

Preserved as reference only — the issues they describe have been resolved:
- `backend-review.md` — Original code review (Jul 2026). All critical/high issues fixed.
- `system-overhaul.md` — Jul 7-8 overhaul: schema fix, pipeline repair, Parquet architecture
- `db-migration-plan.md` — Superseded by Parquet-based architecture
- `schema-fix-recommendations.md` — Schema optimization: composite PK migration
- `price-basis-swap.md` — Steam price basis unification (completed)

## Changelog (`changelog/`)

Dated execution logs for bug fixes, features, and audits. 53 entries, 2026-07-08 to 2026-08-03.

## Other

- `code-review-2026-07-21.md` — **Has open findings.** The pagination double-slice bug it
  reports is still live at `api/routes/items.py:256,267`. Not historical.
- `retrain-optimization-analysis.md` — Per-change training-time estimates and side effects
- `design.md` — Visual design system: OKLCH palette, typography, spacing, components
- `product.md` — Product positioning, users, brand personality, design principles
- `operations.md` — Workflow monitoring: schedules, data flow, troubleshooting
