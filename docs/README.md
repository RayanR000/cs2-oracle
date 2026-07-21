# docs/

## Architecture (`architecture/`)

- `data.md` — Parquet archive, Supabase serving layer, storage breakdown, migration history
- `model.md` — LightGBM forecaster: features, ensembles, regime-switching, accuracy, parameters
- `pipeline.md` — CSGOTrader aggregator: multi-source collection, Parquet storage, coverage

## Reference (`references/`)

- `steam-api.md` — Steam Market API endpoints, rate limits, response format (empirically tested)
- `backfill.md` — CSMarketAPI multi-market backfill: key rotation, priority queue, execution results
- `data-sources.md` — Source quality, freshness, known issues, volume data analysis
- `catalog-build.md` — Phase 1 Steam catalog scrape: rate-limiting strategy, gap repair

## Research (`research/`)

- `feature-engineering.md` — Deep analysis of all feature categories. Most Phase 1-2 items done; some Phase 3-6 speculative
- `accuracy-opportunities.md` — Feature + architecture + training improvements with calibrated impact estimates
- `volume-data.md` — Volume data evaluation: free source in archive, zero predictive lift verified
- `competitor-analysis.md` — Landscape: CSMarketCap, SteamAnalyst, TradeUp Academy, differentiators
- `2026-07-14-remaining-accuracy-improvements.md` — Planning doc: completed + moderate + deeper accuracy items
- `2026-07-18-model-critique.md` — Speed/accuracy audit after column-order-bug fix
- `2026-07-19-feature-contribution-by-horizon.md` — Ablation study: cross-sectional + events by horizon

## Historical (`historical/`)

All historical docs are preserved as reference only — the issues they describe have been resolved:
- `backend-review.md` — Original code review (Jul 2026). All critical/high issues fixed.
- `system-overhaul.md` — Jul 7-8 overhaul: schema fix, pipeline repair, Parquet architecture
- `db-migration-plan.md` — Superseded by Parquet-based architecture
- `schema-fix-recommendations.md` — Schema optimization: composite PK migration
- `price-basis-swap.md` — Steam price basis unification (completed)

## Changelog (`changelog/`)

Dated execution logs for bug fixes, features, and audits. 34 entries covering Jul 2026 development.

## Other

- `design.md` — Visual design system: OKLCH palette, typography, spacing, components
- `product.md` — Product positioning, users, brand personality, design principles
- `operations.md` — Workflow monitoring: schedules, data flow, troubleshooting
- `next-steps-tier1.md` — Historical planning doc (Tier-1 speedups — all items completed)
