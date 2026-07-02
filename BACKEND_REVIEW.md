# Backend Code Review — 2026-07-02

Read-only review of the database, collectors, analytics, scripts, and GitHub Actions workflows. Nothing has been changed. Findings are ranked by severity; the last section covers what your Steam Web API key can (and can't) be used for.

## How the backend works today

GitHub Actions cron jobs are the entire backend runtime, writing to Supabase Postgres (`SUPABASE_DATABASE_URL`); local dev defaults to SQLite (and a stale `backend/cs2_market.db` is committed to the repo). Data sources:

1. **CSGOTrader aggregator** (`prices.csgotrader.app` steam + skinport dumps) — the primary source, every 6 hours, stored as `source='aggregator_sync'`.
2. **Steam Community Market** — unofficial, unauthenticated scraping endpoints (`market/search/render/`, `market/priceoverview/`) with a self-imposed 15–20 s delay. **No Steam Web API call exists anywhere; `settings.steam_api_key` in `backend/config.py:23` is dead config.**
3. **CSFloat listings API** — no key, no rate limiting; likely 4xx-ing in practice (the API generally requires an Authorization key now) and only passes tests because requests are mocked.
4. **cs2.sh / Steam announcements importers** — empty stubs; the actual fallback is a synthetic linear price generator.

Downstream: daily trend analysis (03:00 UTC), daily LightGBM forecast (04:00, full retrain Mondays), weekly item discovery, event correlation, long-term trends, and Sunday pruning/downsampling.

---

## Critical

### 1. Every workflow silently swallows failures (all 7 `.github/workflows/*.yml`)
Steps use `python scripts/run_task.py ... 2>&1 | tee foo.log; EXIT_CODE=$?`. GitHub's default shell has no `pipefail`, so the step status is `tee`'s exit code — always 0. A crashed script logs "✅ completed with exit code: 0", the step passes, and the "Notify on failure" step never fires. **You currently have zero failure visibility.**
Fix: `set -o pipefail` (or `shell: bash`) in every run step; drop the dead `EXIT_CODE=$?` lines.

### 2. Forecaster ignores the item filter (`backend/models/forecaster.py:357–396`)
`predict(item_ids=...)` filters `latest` at line 367 but never uses it; predictions are built from the full dataframe, and `predict_single` returns `results.iloc[0]` — an arbitrary item, not the requested one.

### 3. One shared SQLAlchemy Session across all scheduler jobs (`backend/collectors/pipeline.py:36–43`)
`Session` is not thread-safe; APScheduler jobs can overlap and corrupt session state, and a failed job leaves the session dirty for the next one. Create a fresh `SessionLocal()` inside each `run_*` method (or use `scoped_session`).

### 4. "Daily" features operate on rows, not days (`forecaster.py:102–128`, `pipeline.py:830–832`)
Collection writes ~4 rows/item/day, so `price_lag_1d` is really ~6 h, `price_mean_7d` ≈ 2 days, and `target_7d` isn't 7 days ahead. The same row-vs-day confusion affects the SMA/volatility in `run_feature_computation`. Resample to one row per item per day before feature engineering.

### 5. Fabricated data contaminates analytics and training
- `pipeline.py:239–274`: the historical fallback re-inserts stale prices with `timestamp=now`, creating artificial flat series for items the aggregator missed.
- `free_data_importer.py:52–73`: synthetic "+$0.01/day from $1.00" placeholder series are written into `price_history`, with no dedup on re-run.
- Nothing anywhere filters by `source`, so all of this feeds trend analysis and LightGBM training as real market data.

### 6. No `concurrency` groups or `timeout-minutes` in any workflow
All 7 jobs write the same DB; the default 6-hour timeout plus a slow aggregator/discovery run guarantees overlap with the 03:00/04:00 jobs — duplicate inserts (see #8) and lock contention with pruning. The cron-offset "runs after the aggregator" comments are a race, not a dependency; chain with `workflow_run` triggers instead.

---

## High

### 7. Momentum threshold off by 100× (`analytics/trend_analyzer.py:434–464`)
`change` is in percent units but compared to `MOMENTUM_MIN_CHANGE = 0.05` — any 0.05% move counts as momentum, inflating detected opportunities. Related: the MACD signal line (lines 168–181) is an EMA of a single value, so `histogram == 0` always.

### 8. `price_history` has no uniqueness constraint (migration 0001:91–105)
No `UNIQUE(item_id, timestamp, source)`, so overlapping runs or re-runs after partial failure insert duplicates that skew every downstream calculation. Add the constraint + `ON CONFLICT DO NOTHING`.

### 9. `item_forecasts` exists only in the ORM, not in Alembic
Migrations stop at 0002, but `database.py:168` defines `ItemForecast` and the forecast workflow writes to it. A fresh `alembic upgrade head` environment lacks the table. Add migration 0003.

### 10. Monday's full retrain is thrown away (`price-forecast.yml`)
Models load from repo-committed files in `backend/models/saved_models/`, but the workflow never commits or uploads retrained artifacts — Tue–Sun runs always use stale repo models. Commit them back or store them as a weekly-keyed artifact/cache.

### 11. Prune script hides errors and can't run on SQLite (`scripts/prune_database.py`)
The first-attempt exception at lines 157–166 is discarded with no logging; per-batch failures still exit 0. Raw f-string date interpolation (line 141) and Postgres-only SQL (`to_char`, `::text`, `ANY(...)`) break the local SQLite path the rest of the codebase supports. Also `to_char(timestamp, 'YYYY-WW')` uses non-ISO weeks — buckets split at year boundaries; use `IYYY-IW`.

### 12. Steam name resolution can attach wrong prices (`collectors/steam_market.py:88–156`)
Matching only the first two name tokens can resolve "AK-47 | Redline (Field-Tested)" to a StatTrak or different-wear listing, permanently polluting the DB. Require exact match after normalization. Also `run_daily_collection` walks the full catalog via `search/render` (~240 requests, 80+ min at 20 s pacing) against Steam's most aggressively rate-limited endpoint, and a single failure silently truncates the run yet reports `"status": "success"`.

### 13. Validation layer is dead code
`DataValidator`/`DataCleaner` in `data_validation.py` are never invoked by the pipeline — prices go straight to `price_history` with only a `> 0` check. `validate_price` also passes `NaN`, and booleans validate as volume.

---

## Medium

- **Sub-task failures ignored** — `run_task.py:100–104` prints `analyze_trends.main()`'s failure code but never exits non-zero; `hardcoded venv/bin/alembic` (line 28) always fails first in CI.
- **Notify steps likely can't create issues** — no `permissions:` block in any workflow, and `2>/dev/null || true` hides the `gh issue create` failure. Add `permissions: { issues: write, contents: read }`.
- **Missing indexes for prune query patterns** — `daily_analysis.analysis_date`, `event_impacts.created_at`, `event_correlations.created_at`, `trend_indicators.timestamp` all get weekly sequential scans.
- **Memory-heavy latest-price loading** — `pipeline.py:404–455` loads every history row per item set just to keep the first; use a group-by max-timestamp subquery or window function.
- **Timezone inconsistency** — Steam collector uses naive `datetime.utcnow()`, CSFloat returns tz-aware, csgotrader strips tzinfo; standardize on one helper.
- **Cross-marketplace price mixing** — Skinport prices silently overwrite Steam prices in one merged dict stored under a single `aggregator_sync` source (`csgotrader_aggregator.py:133–157`); also the Skinport parser reads the wrong JSON keys (`last_24h`/`price` vs actual `starting_at`/`suggested_price`), so it likely contributes nothing.
- **`echo=settings.debug` with `debug=True` default** logs every SQL statement in the 6-hourly jobs; the cwd-relative SQLite default silently creates a second empty DB when scripts run from elsewhere.
- **Massive workflow duplication** — all 7 files share ~45 identical lines (checkout/setup/pip/notify/upload). One reusable `workflow_call` workflow with a `task` input would collapse them; also add `cache: pip` (every run reinstalls LightGBM/pandas from scratch).
- **Hardcoded `TRAIN_SPLIT_DATE = "2026-06-15"`** (`forecaster.py:25`) silently degrades as data ages; make it relative. Also the forecaster looks for event type `"game_update"` but the importer writes `"update"`, so that feature is permanently null.

## Low

- Remove `backend/cs2_market.db` from git and fail fast in production if `DATABASE_URL` is unset.
- Hardcoded default `secret_key` with no production guard; `extra="allow"` in config masks env-var typos.
- `steam_market.py:240–259` `get_item_name_id` can never return a value (upstream never populates `name_id`) — dead code.
- Trend results (`trend_score`/`direction`/`confidence`) computed in `run_trend_analysis` are never persisted — log-only work.
- `pipeline.py:835` `if sma_7 or sma_30:` treats a legitimate 0.0 as missing; use `is not None`.
- Retry logic: 429 backoff shares the retry counter with network errors, pacing is applied only once before the retry loop, and the User-Agent "rotation" happens once at construction.
- `price_history.id` is `Integer`; at ~68k rows/day, `BigInteger` is the safer choice.
- Volatility annualized with 252 trading days; CS2 markets trade 365. Magic numbers throughout (0.85/1.20, 999 sentinels, top-2000, 20 s delay) belong in config.

## Top 5 to fix first

1. `pipefail` in all workflows (#1) — everything else is invisible until this is fixed.
2. Concurrency groups + timeouts (#6) and the `price_history` unique constraint (#8) — these explain how bad data enters silently.
3. Stop fabricating data points (#5) or at least tag and filter by `source` in analytics/training.
4. Forecaster item-filter bug (#2) and row-vs-day features (#4) — the ML output is currently unreliable.
5. Persist the Monday retrain (#10).

---

# What your Steam Web API key can do (100k calls/day)

Important caveat first: **the official Steam Web API (`api.steampowered.com`) does not expose Community Market prices.** The market endpoints you're scraping (`priceoverview`, `search/render`) are unofficial community endpoints that ignore API keys — the key won't raise those rate limits or make that scraping legitimate. So the key can't replace your price collection, but it unlocks several things the codebase currently stubs or lacks:

### 1. Game update / event feed — replaces your empty stubs (highest value)
`ISteamNews/GetNewsForApp/v2/?appid=730` returns CS2 patch notes and announcements. Your `SteamAnnouncementsImporter` and `CS2ShClient` are empty stubs, and the event-correlation analysis plus the forecaster's event features (`days_since_event`, currently permanently null due to the type mismatch) depend on exactly this data. One call per day fills the `events` table with real update/case-release events.

### 2. Player count as a demand feature
`ISteamUserStats/GetNumberOfCurrentPlayers/v1/?appid=730` gives live CS2 concurrent players. Polled hourly (24 calls/day) it becomes a genuinely predictive demand-side feature for the LightGBM models — skin prices correlate with player activity, especially around major updates and events.

### 3. Item metadata and icons
`ISteamEconomy/GetAssetClassInfo/v1` (appid 730) returns official item metadata — icons, types, descriptions — useful for enriching the `items` table and the frontend without scraping listing pages. Note `GetAssetPrices` only covers store-priced assets, not market items, so it's not a price source.

### 4. Inventory valuation feature (future product surface)
`IEconService`/inventory endpoints plus `ISteamUser/GetPlayerSummaries` let you build "value my inventory" — resolve a user's SteamID, pull their CS2 inventory, and price it against your DB. That's a natural product feature and a common ask for market-analyzer tools. (CS2 inventory contents come from a community endpoint too, but profile resolution/vanity-URL → SteamID64 needs the key.)

### 5. Budget reality check
Your realistic usage is tiny relative to 100k/day: ~24 player-count polls + a few news fetches + occasional metadata batches ≈ under 100 calls/day. The budget is not the constraint — the constraint is that official endpoints simply don't cover market prices. If you want a legitimate high-volume price source, the better use of effort is fixing the CSFloat collector (get a CSFloat API key, add rate limiting) and correcting the Skinport JSON parsing, both flagged above.
