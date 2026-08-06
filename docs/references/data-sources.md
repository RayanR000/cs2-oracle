# Data Source Audit & Plan

## Current Sources

| Source | Type | Interval | Freshness | Auth | Status |
|---|---|---|---|---|---|
| CSGOTrader aggregator | JSON API | Daily (23:00 UTC) | 24h avg of Steam sales (stale) | None | **Active (primary — the only live collector)** |
| Steam Market supply scraper | Burst scrape | Not running | Live sell_listings count | None | **Dormant** — workflow deleted (Steam 429s runner IPs); features also excluded by `FEATURE_GROUP_ALLOWLIST` |
| CSFloat API | REST API | Not running | Live listings | API key not configured | **Degraded** |
| Steam Web API | REST API | Manual only | Item schema/icons | STEAM_API_KEY (optional) | **Not used by any pipeline** |
| Skinport (via aggregator) | JSON API | Daily | Reads `starting_at` correctly | None | **Active** (fixed — `csgotrader_aggregator.py:191-192, 314-317`) |
| Skinport (direct API) | REST API | N/A | Live `/v1/items` | None | **Available, unimplemented** — API 200s with `Accept-Encoding: br` |
| cs2.sh archive | API stub | N/A | Not implemented | CS2SH_API_KEY | **Stub** |
| **HF CS2 Dataset (idomanteu)** | **Parquet (Hugging Face)** | **Imported once** | **Hourly BUFF/CSFloat/YouPin, Mar 22 – Apr 15 2026** | **None (CC BY 4.0)** | **✅ Active (merged to archive 2026-07-20)** |
| Steam Announcements | Stub | N/A | Not implemented | None | **Stub** |
| Synthetic demo | Generated | Dev only | Fake | None | **Dev only** |
| Steam `priceoverview` | Undocumented endpoint | Per-item | 24h sales volume, lowest/median price | None | **Not integrated** |
| **Steam market listing pages** | **SSR HTML scrape** | **One-shot backfill** | **Daily median + volume back to 2013** | **None (no cookie)** | **⚠️ Staged** — 262 items collected; this IP soft-blocked 2026-08-05 |
| BUFF163 git archive (atalantus) | JSON dump (24 MB xz) | One-shot | Daily CNY min-listing, 2021-07 → 2024-02 | None (unlicensed) | **Evaluated, declined** |
| CSFloat `/api/v1/history/…/graph` | REST (undocumented) | Per-item | Daily completed-sale avg + count, from 2020-04 | None | **Not integrated** |
| CSMarketCap API | GraphQL + REST | Bulk (all items in 1 call) | Trade volume (24h/7d/30d/90d), listings, buy orders | JWT token | **Not integrated** ($9.99/mo) |

## CSGOTrader Accuracy Issues

- `steam.json` is a **rolling 24h average** of completed Steam Market sales, NOT a live price
- Lags significantly on volatile items (new cases, sticker releases, event spikes)
- CSGOTrader's `csgotrader` source records `volume=None` (→0) — it carries no trade volume (verified: `prices.csgotrader.app/latest/csgotrader.json` returns only `price` + `doppler`). The archive's `aggregator_sync` Steam backfill does carry real trade volume, so liquid vs illiquid items *can* be distinguished for backfilled items.
- No freshness metadata in the JSON dump — can't detect stale/failed upstream
- `data_validation.py` has outlier/anomaly checks but they are NEVER called in the pipeline
- Historical fallback re-inserts stale prices with `timestamp=now`, but the rows are relabelled `historical_fallback:<source>` (`pipeline.py:39-43`) and stale items are tracked separately (`pipeline.py:183-238`) — so fallback rows *are* distinguishable downstream. Treat this as a freshness caveat, not silent corruption.

## Supply Scraper (Steam sell_listings) — DELETED

**Added 2026-07-15, removed 2026-08.** Do not treat this as a live source.

- **Why it died:** hosted GitHub runners are 429'd by `steamcommunity.com` on the *first* request. `supply-scraper.yml` was deleted (commit 0288568); `supply_snapshots` is frozen at 35,037 rows.
- **It fed nothing anyway:** `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` (`models/forecaster.py:222`) discards supply features before training. The forecaster has never consumed supply-depth features in production.
- **Code still present but unreachable from CI:** `backend/collectors/supply_scraper.py`, entry at `backend/scripts/run_supply_scraper.py`. Runnable manually from a residential IP only.
- **Original design (for reference):** burst scrape of `steamcommunity.com/market/search/render/` (public, no auth), 20 rapid requests → 30s pause, ~3,400 pages of 10 items → ~115 min for the full catalog.

### Skinport
The direct Skinport API (`/v1/items`) is **alive** — the historical 403s were caused by a missing `Accept-Encoding: br` request header, not Cloudflare Bot Management. No `api.skinport.com` client exists yet; see the source table above. Skinport data arriving via the CSGOTrader aggregator is correct (`starting_at`, not `last_24h`).

## Deduplication strategy
- Only insert price row if value actually changed vs previous row
- Without dedup at 5-min intervals: ~3.9M rows/day (200GB/year — not viable)
- With dedup at 30-min intervals: ~65-130K rows/day (~10-18 MB/day, ~3.5-6.5 GB/year)
- Fits comfortably in Supabase Pro (8GB)

## Steam priceoverview as fallback
- **Not implemented.** There is no Steam Market collector in the repo (`backend/collectors/` has no `steam_market.py`), so this is greenfield work, not a wiring job.
- Would cover items the aggregator misses
- Rate-limited (~1 req/sec) — fine for gap-filling, but see `steam-api.md`: hosted CI runners are 429'd immediately, so this can only run from a residential IP.

## Volume Data Status (audited 2026-07-16, corrected 2026-07-16)

Volume **is** present in the Parquet archive — and it is **not** limited to a 90-day window.

- **Coverage:** 9,833,838 rows (**88.65%** of all 11,092,908 price rows) carry non-zero `volume`, spanning **2013-08-14 → 2026-03-29** across **5,542 unique items**.
- **Source label:** these rows are tagged **`aggregator_sync`** in the archive. Older analysis scripts and the backfill DB still call this `STEAMCOMMUNITY` — same data; the `source` column was added later and rows without it were defaulted to `aggregator_sync` (`append_to_parquet.py:119` for the legacy CSV path, `:191` for the schema migration of existing Parquet).
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

## Hugging Face CS2 Dataset (merged 2026-07-20)

The [HF CS2 Historical Item Price Dataset](https://huggingface.co/datasets/idomanteu/cs2-historical-item-prices-hourly-march-april-2026) (CC BY 4.0) was merged into the Parquet archive to fill the 2026 data gap and expand item coverage.

### What it contains

| Field | Detail |
|-------|--------|
| Source | `idomanteu/cs2-historical-item-prices-hourly-march-april-2026` |
| Format | Parquet (zstd), 668 MB raw, 69.2M rows |
| Period | 2026-03-22 → 2026-04-15 (25 days, hourly) |
| Items | 32,848 unique `market_hash_name` |
| Markets | BUFF (`aggregator_buff163`), CSFloat (`aggregator_csfloat`), YouPin (`aggregator_youpin`) |
| Fields | OHLC ask/bid, ask/bid volume, sample count |
| License | CC BY 4.0 — free for commercial use with attribution |

### What was merged

| Scope | Days | Items added per day | Sources |
|-------|:----:|:-------------------:|:-------:|
| Gap fill (Mar 30 – Apr 15) | **17** | ~32,500 (previously 0) | 3 (buff/csfloat/youpin) |
| Overlap expansion (Mar 22–29) | 8 | ~32,400 (was 504–2,724) | 4 (buff/csfloat/youpin + existing aggregator_sync) |

### Before vs After (2026 Parquet archive)

| Date range | Before | After |
|------------|--------|-------|
| Mar 22 | 2,724 items, 1 source | 32,437 items, 4 sources |
| Mar 30 – Apr 15 | **zero data** | ~32,500 items/day, 3 sources |
| Jul 11+ | 41,125 items, 11 sources | unchanged |

The remaining gap (**Apr 16 – Jul 8, 84 days**) is still unfilled for non-backfilled items.

### Merge script

`backend/scripts/merge_hf_dataset.py` — standalone script that:
1. Downloads the HF Parquet file (cached at `/tmp/cs2_listing_prices_hourly.parquet`)
2. Maps `market_hash_name` → `item_slug`, `bucket` → `day`, `close_ask` → price
3. Aggregates hourly → daily OHLCV per `(item_slug, day, source)`
4. Appends to `prices-YYYY.parquet` and `snapshots-YYYY.parquet` using the same dedup logic as `append_to_parquet.py`

Usage: `python scripts/merge_hf_dataset.py --out-dir ..`

## Steam market listing pages — free logged-out history (evaluated 2026-08-05)

The conventional wisdom, repeated in every community thread, is that Steam price
history requires a logged-in session. That is true of `/market/pricehistory/`
(**HTTP 400** without a cookie), but **not** of the listing page:

```
GET https://steamcommunity.com/market/listings/730/<market_hash_name>   # follow the 302
```

Its dehydrated React-Query blob embeds the same series — `{time, price_median,
purchases}`, where `purchases` is real daily traded volume — with **no cookie, no
key, back to 2013**. Pages carry several wear/StatTrak variants at once, so
harvesting all of them yields **~4 items per request**.

Implementation: `backend/scripts/backfill_steam_listing_history.py`. Targets
non-gated, name-keyed items; writes to a staging SQLite
(`runtime/steam_listing_history.db`) and never touches prod or the archive.

### Two traps, both hit and both fixed

**1. Price basis — divide by 1.1607.** The page quotes the **buyer** price
(Steam's fee included); the archive's `aggregator_sync` rows store **net**.
Across 30,875 overlapping pre-2024 rows the ratio is constant: median 1.1607,
p10 1.1565, p90 1.1656, within-item CV 0.0021.

| | median abs diff | correlation |
|---|---|---|
| Raw | 16.08% | 1.0000 |
| ÷1.1607 | **0.038%** | **1.000000** |

Volume needs no correction (77.8% exact match before normalisation).

**2. Steam soft-blocks without a 429.** Instead of an error it serves HTTP
**200** with a stripped ~250 KB shell: no 302, zero occurrences of
`pricehistory`, no `Retry-After`. A 300-request run logged *0 failures, 0 429s
and "80% EMPTY"* while being throttled for most of it — the
collectors-fail-silently shape. Detection now in `classify()`: a real hit
redirects **and** contains `pricehistory`; anything else under 300 KB with no
redirect is a block, not an empty item. A **canary** (a liquid item that must
have history) runs before the run and every 50 requests, and aborts rather than
marking items done during a block.

### Rate limiting — what is actually known

Valve publishes nothing. The community figure is **~20 requests/minute**, with an
**IP-based** limiter added to Market/inventories in **October 2022**.

Measured here: a 20-request burst at 1.75 req/s succeeded fully (it merely
drained the bucket), but a sustained run at 2.5 s + 0.8 s jitter ≈ **20.7
req/min** — right on the line — tripped the block, and then issued ~250 further
requests while blocked. The result was **IP-scoped and long-lived: still blocked
2.4 h later with zero traffic**, identical across 6 probes with fresh and warmed
sessions. Fetching the same URL **from a different IP returned the fully
populated page**, confirming the flag is on the caller, not a Steam-wide change.

The exact bucket depth and refill period were **not** isolated — doing so means
deliberately re-tripping a flagged IP. Practical guidance instead:

- Use `--delay 8` or slower (≤8 req/min), well under the ~20/min line.
- Treat the first stripped page as a full stop. Never request during a block —
  that is what turned a timeout into a multi-hour penalty.
- Probe recovery with a single request (`--limit 1 --resume`), hours apart.
- Do **not** run from GitHub Actions; Steam 429s runner IPs.

### Measured value

262 items / 528,573 rows collected before the block, back to 2013-08-15. Unlike
the depth-only candidates below, this source *lowers* avg rows/item and so
**raises** training breadth:

| | items | rows/item | `target_items` @700k |
|---|---|---|---|
| Current gated pool | 5,542 | 1,341 | 521 |
| + sample harvested | 5,770 | 1,321 | 529 |
| Extrapolated to all 29,933 targets | ~31,590 | 923 | **758 (+45%)** |

Treat 5.7× breadth as an upper bound — the sample is biased toward items that
resolved. Note also that no extra data can move accuracy while the training row
budget binds; the argument for this source is a training universe that matches
what `predict()` scores, not row count.

## Bulk historical sources evaluated and declined (2026-08-05)

| Source | Free? | Content | Why declined |
|---|:--:|---|---|
| [atalantus/buff-price-history-archive](https://github.com/atalantus/buff-price-history-archive) | Yes | **21,954 items, 15.4M daily rows, 2021-07 → 2024-02**, CNY×100, one 24 MB xz | Depth-only: adds 1.66M in-window gated rows, pushing rows/item 1,341 → 1,641 and `target_items` **521 → 426 (−18%)**. Needs 2021–24 CNY/USD rates the archive lacks. No LICENSE. Fetch via `raw.githubusercontent.com` — the LFS media URL 404s. |
| CSFloat `/api/v1/history/<name>/graph` | Yes | Daily completed-sale avg + count from 2020-04, no auth (1,898 rows for AK Redline) | Viable and unblocked; simply not integrated yet. Extends the existing `aggregator_csfloat` label backwards. |
| cs2.sh | No | Steam daily back to 2013 | No free tier; history is the $200/mo Scale plan and only reaches back to 2025-12-24. |
| CSGOSKINS.GG | No | 36 markets, 38.5K items | €179/mo minimum, 90-day history. |
| SteamAnalyst | Partly | 30+ markets | Free tier is 100 req/day — unusable in bulk. |
| Wayback Machine (`prices.csgotrader.app`) | Yes | — | `steam.json` has **zero** captures; only 4 stray snapshots of older filenames. Not a time series. |
| `HilliamT/scm-price-history` | Yes | The classic `var line1=` scrape | Dead — Steam moved to SSR. The React blob documented above is its replacement. |

## Quality gaps
- Wire `data_validation.py` checks into the pipeline — it is still dead code (only importers are `collectors/__init__.py:1` and `tests/test_data_validation.py:6`; `pipeline.py:160` validates `price > 0` only)
- Historical fallback still emits flat-line rows with `timestamp=now`; they are labelled and tracked, but nothing downstream *excludes* them yet
