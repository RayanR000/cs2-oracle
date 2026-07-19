# docs/

## Architecture (`architecture/`)

- `data.md` — Data architecture: Parquet archive, Supabase serving layer, schema changes, data flow
- `model.md` — ML model: LightGBM quantile regression, features, training, accuracy, audit history
- `pipeline.md` — Aggregator: multi-source collection from CSGOTrader, storage strategy, coverage

## Reference (`references/`)

- `steam-api.md` — Steam Market API endpoints, rate limits, response format, catalog coverage
- `backfill.md` — CSMarketAPI multi-market backfill: architecture, key rotation, execution results
- `data-sources.md` — Data source audit: quality, freshness, known issues per source
- `catalog-build.md` — Phase 1 market catalog: Steam scraping, rate limiting, gap repair

## Research (`research/`)

- `feature-engineering.md` — Deep analysis of current + 115 planned features with implementation plan
- `accuracy-opportunities.md` — Categorized opportunities: features, architecture, training, data quality
- `volume-data.md` — Volume data source evaluation: CS2Cap, SteamWebAPI, CSMarketAPI, recommendations
- `remaining-accuracy-improvements.md` — Master tracker of all remaining opportunities with completion status

## Historical (`historical/`)

- `backend-review.md` — Original code review (Jul 2026). All critical/high issues resolved.
- `system-overhaul.md` — Session notes from the full system overhaul (2026-07-07 → 07-08)
- `db-migration-plan.md` — Database cleanup and historical data migration plan (superseded by Parquet architecture)
- `schema-fix-recommendations.md` — Schema optimization recommendations, composite PK migration
- `price-basis-swap.md` — Steam price basis swap: unified pricing from market.csgo to Steam basis

## Changelog (`changelog/`)

Dated execution logs, bug fixes, and audit findings. See `changelog/` directory for the full list (28 entries).

## Root-level docs moved here

- `design.md` — Visual design system: OKLCH palette, typography, spacing, components
- `product.md` — Product positioning, users, brand personality, design principles
- `operations.md` — Workflow monitoring: schedules, data flow, troubleshooting
