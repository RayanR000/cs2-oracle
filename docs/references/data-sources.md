# Data Source Audit & Plan

## Current Sources

| Source | Type | Interval | Freshness | Auth | Status |
|---|---|---|---|---|---|
| CSGOTrader aggregator | JSON API | Every 6h | 24h avg of Steam sales (stale) | None | **Active (primary)** |
| Steam Market (scraped) | Web scrape | Daily | Live snapshot but rate-limited | Cookies (for pricehistory) | **Active (secondary)** |
| Steam Market supply scraper | Burst scrape | Daily | Live sell_listings count (burst-limited) | None | **Active** (output unused — supply depth **dropped** from accuracy roadmap 2026-07-16) |
| CSFloat API | REST API | Not running | Live listings | API key not configured | **Degraded** |
| Steam Web API | REST API | Manual only | Item schema/icons | STEAM_API_KEY | **Active (manual)** |
| Skinport (via aggregator) | JSON API | Every 6h | Broken — reads wrong keys (`last_24h` instead of `starting_at`) | None | **Broken** |
| Skinport (direct API) | REST API | N/A | Cloudflare 403 — unavailable server-side | None | **Dead** |
| cs2.sh archive | API stub | N/A | Not implemented | CS2SH_API_KEY | **Stub** |
| Steam Announcements | Stub | N/A | Not implemented | None | **Stub** |
| Synthetic demo | Generated | Dev only | Fake | None | **Dev only** |
| Steam `priceoverview` | Undocumented endpoint | Per-item | 24h sales volume, lowest/median price | None | **Not integrated** |
| CSMarketCap API | GraphQL + REST | Bulk (all items in 1 call) | Trade volume (24h/7d/30d/90d), listings, buy orders | JWT token | **Not integrated** ($9.99/mo) |

## CSGOTrader Accuracy Issues

- `steam.json` is a **rolling 24h average** of completed Steam Market sales, NOT a live price
- Lags significantly on volatile items (new cases, sticker releases, event spikes)
- CSGOTrader's `csgotrader` source records `volume=None` (→0) — it carries no trade volume (verified: `prices.csgotrader.app/latest/csgotrader.json` returns only `price` + `doppler`). The archive's `aggregator_sync` Steam backfill does carry real trade volume, so liquid vs illiquid items *can* be distinguished for backfilled items.
- No freshness metadata in the JSON dump — can't detect stale/failed upstream
- Skinport data merged into same dict but broken (reads `last_24h`/`price` keys that don't exist)
- `data_validation.py` has outlier/anomaly checks but they are NEVER called in the pipeline
- Historical fallback re-inserts stale prices with `timestamp=now` — downstream tools see fake fresh data

## Supply Scraper (Steam sell_listings)

**Added 2026-07-15.** Daily collector for supply-side data (sell_listings count).

- **Source:** `steamcommunity.com/market/search/render/` (public, no auth)
- **Strategy:** Burst scrape (20 rapid requests → 30s pause). Full catalog scan (~3,400 pages of 10 items each) → ~115 min.
- **Coverage:** 34K+ items on CS2 Steam Market, filtered to ~5.5K tracked items
- **Storage:** `supply_snapshots` table in Supabase, consumed by forecaster for supply-depth features
- **Runner:** GitHub Actions daily at 22:00 UTC (`supply-scraper.yml`), 120-min timeout
- **Code:** `backend/collectors/supply_scraper.py`, entry at `backend/scripts/run_supply_scraper.py`

### Skinport (dead)
The Skinport direct API (`/v1/items`) is now behind Cloudflare Bot Management (403 on server-side requests). Removed from the scraper. The CSGOTrader aggregator still references Skinport data but it reads the wrong JSON keys so it contributes nothing anyway.

## Deduplication strategy
- Only insert price row if value actually changed vs previous row
- Without dedup at 5-min intervals: ~3.9M rows/day (200GB/year — not viable)
- With dedup at 30-min intervals: ~65-130K rows/day (~10-18 MB/day, ~3.5-6.5 GB/year)
- Fits comfortably in Supabase Pro (8GB)

## Steam priceoverview as fallback
- Already coded in `steam_market.py:294`
- Covers items Skinport misses
- Rate-limited (~1 req/sec) — fine for gap-filling

### Items to fix in aggregator (`csgotrader_aggregator.py:150`)
- Tag each price with its source marketplace
- Skinport JSON parsing reads wrong keys (`last_24h`/`price` vs actual `starting_at`/`suggested_price`) — but Skinport is dead anyway

## Volume Data Status (audited 2026-07-16, corrected 2026-07-16)

Volume **is** present in the Parquet archive — and it is **not** limited to a 90-day window.

- **Coverage:** 9,833,838 rows (**88.65%** of all 11,092,908 price rows) carry non-zero `volume`, spanning **2013-08-14 → 2026-03-29** across **5,542 unique items**.
- **Source label:** these rows are tagged **`aggregator_sync`** in the archive. Older analysis scripts and the backfill DB still call this `STEAMCOMMUNITY` — same data; the `source` column was added later and rows without it were defaulted to `aggregator_sync` (`append_to_parquet.py:219`).
- **Origin:** a Steam price-history backfill. `scripts/backfill_ssr_history.py` pulls `steamcommunity.com/market/pricehistory/` (which returns daily traded volume); the data was merged into the archive via `append_to_parquet.py` (the legacy `--backfilled-csv` path relabels Steam backfill rows to `aggregator_sync`).
- **Per-year:** 2013–2025 are ~100% volume-populated; 2026 is partial (24.3% — only the `aggregator_sync` subset has volume; the live aggregator sources still record `volume=0`).

| Time Period | Volume Source | Status |
|---|---|---|
| 2013 → 2025 | `aggregator_sync` (Steam backfill) | ✅ Non-zero volume, full years |
| 2026 (Jan–Mar) | `aggregator_sync` | ✅ Partial (24% of 2026 rows) |
| 2026 (Apr+ live aggregator) | csgotrader / skinport / etc. | ❌ `volume=0` (those sources don't collect volume) |
| CSMarketAPI backfill DB | `csmarketapi.db` | ❌ Not present on disk (historical) |

### Does volume improve predictions? — No (verified 2026-07-16)

Tested on the volume-rich window (2023–2025, 4.47M samples with `volume>0`):

- Every volume feature correlates with **next-day** and **7-day** forward returns at **|r| < 0.002** — statistical noise.
- Volume also fails to predict move *magnitude* (`|fwd_return|`).
- The only real predictive signal in the data is **price momentum**: `corr(return_7d, fwd_return_7d) = +0.0796`.

**Conclusion:** volume features will **not** improve forecast accuracy. Volume's value in this stack is **data quality / confidence** — the existing `detect_market_manipulation` filter (`data_validation.py`) and `volume_price_conf` liquidity weighting — not predictive power. Sourcing additional volume (CSMarketCap, Steam `priceoverview`) is therefore **not** justified by prediction accuracy; it only helps liquidity/confidence weighting and fills the ~34K items that still lack volume.

### Volume data sources evaluated (2026-07-16)

| Source | Cost | Bulk? | Trade Volume Fields | Coverage |
|--------|:----:|:-----:|:-------------------|:--------:|
| **Steam price-history backfill (already in archive)** | Free | n/a (historical) | daily trade volume | 5,542 items, 2013–2026 |
| Steam `priceoverview` | Free | No (per-item) | 24h sales count | All Steam items (slow) |
| CSMarketCap API (Standard) | **$9.99/mo** | ✅ 1 call = all items | `last_24h/7d/30d/90d`, `avg_daily_volume` | All Steam items |
| SteamWebAPI Item Small | €15/mo | ✅ 1 call = all items | `sold24h/7d/30d/90d` | All Steam items |
| CS2Cap Pro | $79/mo | ✅ batch (1K items) | `sales_1d/7d/30d` | All items, 40+ markets |
| Pricempire Standard | $99.90/mo | No (per-item) | trade count metas | All items |
| cs2.sh Developer | $75/mo | ✅ bulk endpoint | ask_volume (listing count, not trade vol) | 6 markets |

**Verdict:** a free, bulk trade-volume source already exists *inside the archive* — the Steam price-history backfill (`aggregator_sync`) — for 5,542 items. Paid sources (CSMarketCap $9.99/mo, SteamWebAPI €15/mo) would only extend coverage to more items and keep recent days fresh; they do **not** add predictive signal.

## Quality gaps
- Wire `data_validation.py` checks into the pipeline
- Add stale-data detection (compare timestamps)
- Stop creating fake flat-line data via historical fallback
