# Data Source Audit & Plan

This file catalogues **where data comes from**. For what is actually on disk and how much
of the market it covers — item counts, history depth, per-source spans, field
availability, label coverage — see **`data-inventory.md`**. The acquisition priorities
derived from it are in `../changelog/2026-08-06-data-acquisition-ranking.md`.

## Current Sources

| Source | Type | Interval | Freshness | Auth | Status |
|---|---|---|---|---|---|
| CSGOTrader aggregator | JSON API | Daily (23:00 UTC) | 24h avg of Steam sales (stale) | None | **Active (primary — the only live collector)** |
| Steam Market supply scraper | Burst scrape | Not running | Live sell_listings count | None | **Dormant** — workflow deleted (Steam 429s runner IPs); features also excluded by `FEATURE_GROUP_ALLOWLIST` |
| CSFloat API | REST API | Not running | Live listings | API key not configured | **Degraded** |
| Steam Web API | REST API | Manual only | Item schema/icons | STEAM_API_KEY (optional) | **Not used by any pipeline** |
| Skinport (via aggregator) | JSON API | Daily | Reads `starting_at` correctly | None | **Active** (fixed — `csgotrader_aggregator.py:191-192, 314-317`) |
| Skinport (direct API) | REST API | Daily (attempted) | Live `/v1/items` `quantity` | None | **Wired, blocked** — `collectors/supply_depth.py` calls it daily; returned **HTTP 403 (WAF)** on 2026-08-06 and contributed 0 rows. Two distinct failure modes — see the Skinport section below |
| cs2.sh archive | API stub | N/A | Not implemented | CS2SH_API_KEY | **Stub** |
| **HF CS2 Dataset (idomanteu)** | **Parquet (Hugging Face)** | **Imported once** | **Hourly BUFF/CSFloat/YouPin, Mar 22 – Apr 15 2026** | **None (CC BY 4.0)** | **✅ Active (merged to archive 2026-07-20)** |
| Steam Announcements | Stub | N/A | Not implemented | None | **Stub** |
| Synthetic demo | Generated | Dev only | Fake | None | **Dev only** |
| Steam `priceoverview` | Undocumented endpoint | Per-item | 24h sales volume, lowest/median price | None | **Not integrated** |
| **Steam market listing pages** | **SSR HTML scrape** | **One-shot backfill** | **Daily median + volume back to 2013** | **None (no cookie)** | **⚠️ Staged** — 262 items / 528,573 rows from 47 requests (**5.6 series/request**); this IP soft-blocked since 2026-08-05 and **still blocked 2026-08-08**, so the decay bound is ≥3 days, not ≥5 h. The `--min-price` crash is **fixed 2026-08-08** (32,374 targets / 27,785 at ≥$1); egress is now the only blocker — see `../research/2026-08-07-next-steps.md` **5d** |
| kieranpoc Kaggle sale dump | Kaggle dataset (901 MB) | One-shot | Steam price + sale counts back to 2013, **frozen at 2024-05-04** (uploaded 2024-06-15) | None (CC BY-NC-SA 4.0) | **Declined 2026-08-08** — supplies nothing for 2024-06 → 2026-08, the only window that needs it; the listing pages carry the same `purchases` field live |
| BUFF163 git archive (atalantus) | JSON dump (113 MB via Git LFS) | One-shot | Daily CNY min-listing, 2021-07 → 2024-02; listing counts only after 2023-01-25 | None (unlicensed) | **Declined 2026-08-08** — the listing-count window is 359 days (~2 folds) and cannot splice to live supply depth, which starts 2026-08-06 |
| CSFloat `/api/v1/history/…/graph` | REST (undocumented) | Per-item, **500 req/day** | Daily completed-sale avg + count, from 2020-04 | None | **Measured and declined 2026-08-06** — coverage passes, the basis feature is null; see `../changelog/2026-08-06-csfloat-basis-refuted.md` |
| CSMarketCap API | GraphQL + REST | Bulk (all items in 1 call) | Trade volume (24h/7d/30d/90d), listings, buy orders | JWT token | **Not integrated** ($9.99/mo) |
| **Skinport `/v1/sales/history`** | REST API | **Daily** (1 call, 1.9 s / 20.4 MB) | min/max/avg/median/volume for 24h, 7d, 30d, 90d — **retroactive on first pull** | None (needs `Accept-Encoding: br`) | **✅ Wired 2026-08-08** — `collectors/sales_volume.py` → `volume-YYYY-MM.parquet`. The pipeline's only real trade-volume feed. 36,004 items; 23,533 of the ≥$1 cohort (89.0%). **Unverified from a CI runner** — see below |
| **lis-skins full export** | JSON dump | **Daily** (1 call, 364.5 s / 173 MB) | 2.3M individual listings: `price`, `created_at`, `item_float` — reduced to per-item ask ladder + listing age | None | **✅ Active** — `collectors/supply_depth.py`, 23,879 items on 2026-08-06; 17,286 of the ≥$1 cohort. Largest marginal contributor (+1,428 items over the other feeds). Untested from a GitHub runner |
| **market.csgo.com `/api/v2/prices/USD.json`** | REST API | **Daily** (1 call, 0.6 s / 2.6 MB) | `volume` — a live **listing count**, not trade volume (verified 4 ways, see below) | None | **✅ Active** — 27,570 items on 2026-08-06; 17,596 of the ≥$1 cohort. Untested from a GitHub runner |
| **Waxpeer `/v1/prices?game=csgo`** | REST API | **Daily** (1 call, 0.6 s / 4.9 MB) | `count` (listing count), `min` in millicents | None | **✅ Active** — 22,039 items on 2026-08-06; 15,016 of the ≥$1 cohort. Marginal value only **+117 items** over the other feeds; kept on cost, not on signal |
| **Bitskins `/market/insell/730`** | REST API | **Daily** (1 call, 0.5 s / 1.4 MB) | `quantity` (listing count), `price_min` in millicents | None | **✅ Active** — 10,920 items on 2026-08-06; 4,899 of the ≥$1 cohort |
| `somespecialone/steam-item-name-ids` | GitHub JSON | One-shot (pushed 2026-08-03) | 26,935 `market_hash_name` → `item_nameid` — unblocks `itemordershistogram` lookups | None | **Not integrated** — Steam-hosted consumer, so residential IP only (see IP-class section) and per-item |
| `ByMykel/CSGO-API` | Raw GitHub JSON | One-shot (12 dumps, cached under `runtime/bymykel/`) | 45,362 items — rarity (`rarity_meta` token + `rarity_meta_rank`), item age, float range, StatTrak/souvenir, crate + collection | None | **Ingested** by `scripts/ingest_bymykel_metadata.py` → `item-metadata-bymykel.parquet`. Features **refuted** (2026-08-06), but it is the rarity fill source for `item-metadata.parquet`, taking coverage 50.6% → 99.9% (2026-08-07) |
| Steam `ISteamNews/GetNewsForApp` | REST API | **Weekly** (~2 s, 4 pages) | **1,752 entries back to 2012-03-16** (439 official) — free CS2 event calendar | None | **✅ Scheduled 2026-08-08** in `event-correlation-analysis.yml`, before its only consumer. `scripts/ingest_steam_news.py` writes **two** tables: `event-calendar.parquet` (date-level panel) and `event-news.parquet` (per-event), and `scripts/sync_events_from_news.py` upserts Valve posts into the `events` DB table |
| **iflow BUFF backfill** (`EricZhu-42/SteamTradingSiteTracker-Data`) | JSON dump via `api.iflow.work` | One-shot backfill (12h dumps) | BUFF ref price (**CNY**), `count_in_24` (**live 24h trade volume**), Steam order-book depth, buff buy/sell counts; **2022-04-18 → 2026-05-20**, ~16.6k CS+Dota items/dump | None | **🔬 Candidate (Phase 0, 2026-08-16)** — free ~4yr history; `hash_name` = `item_slug` verbatim, CNY→USD via `exchange-rates-history.parquet`. Two reasons: `count_in_24` is the **live volume series** the open volume lead needs, and 4yr history multiplies backtest episodes ~10× to re-test the underpowered depth/count nulls. See `../research/2026-08-16-refutation-power-tiers-and-iflow-backfill.md` |

## CSGOTrader Accuracy Issues

- `steam.json` is a **rolling 24h average** of completed Steam Market sales, NOT a live price
- Lags significantly on volatile items (new cases, sticker releases, event spikes)
- CSGOTrader's `csgotrader` source records `volume=None` (→0) — it carries no trade volume (verified: `prices.csgotrader.app/latest/csgotrader.json` returns only `price` + `doppler`). The archive's `aggregator_sync` Steam backfill does carry real trade volume, so liquid vs illiquid items *can* be distinguished for backfilled items.
- **No aggregator feed carries a volume field at all.** Re-probed live 2026-08-08:
  `steam.json` returns `last_24h/7d/30d/90d` *prices*, `buff163.json` returns
  `starting_at`/`highest_order`, `csgotrader.json` returns `price` + `doppler`.
  `csgotrader_aggregator.py` contains no reference to `volume` because there is nothing to
  parse. The daily collector has never collected volume — this is an absence, not a
  regression. Real sale counts now arrive on a separate path
  (`collectors/sales_volume.py` → `volume-YYYY-MM.parquet`).
- **The pipeline used to write `0` for that absence; since 2026-08-08 it writes NULL.**
  `pipeline.py`'s `VOLUME_NOT_OBSERVED` and `append_to_parquet.py`'s `_sum_observed`
  (a `min_count=1` sum, so an all-absent group stays NA). This is fix-forward only —
  stored rows through 2026-08-08 keep their fabricated zeros.
- No freshness metadata in the JSON dump — can't detect stale/failed upstream
- `data_validation.py` has outlier/anomaly checks but they are NEVER called in the pipeline
- Historical fallback re-inserts stale prices with `timestamp=now`, but the rows are relabelled `historical_fallback:<source>` (`pipeline.py:39-43`) and stale items are tracked separately (`pipeline.py:183-238`) — so fallback rows *are* distinguishable downstream. Treat this as a freshness caveat, not silent corruption.

## Supply Scraper (Steam sell_listings) — DELETED

**Added 2026-07-15, removed 2026-08.** Do not treat this as a live source. **Replaced 2026-08-06** by `collectors/supply_depth.py`, which pulls four non-Steam marketplace feeds daily inside `aggregator-update.yml` and writes `price-archive/supply-YYYY-MM.parquet`. `models/forecaster.py::_fetch_supply_snapshots` now reads that archive instead of the `supply_snapshots` table. The allowlist point below is unchanged: **the forecaster still consumes no supply-depth feature in production, and no lift has been measured.** See `../changelog/2026-08-06-supply-depth-collector.md`.

- **Why it died:** hosted GitHub runners are 429'd by `steamcommunity.com` on the *first* request. `supply-scraper.yml` was deleted (commit 0288568); `supply_snapshots` is frozen at 35,037 rows.
- **It fed nothing anyway:** `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` (`models/forecaster.py:222`) discards supply features before training. The forecaster has never consumed supply-depth features in production.
- **Code still present but unreachable from CI:** `backend/collectors/supply_scraper.py`, entry at `backend/scripts/run_supply_scraper.py`. Runnable manually from a residential IP only.
- **Original design (for reference):** burst scrape of `steamcommunity.com/market/search/render/` (public, no auth), 20 rapid requests → 30s pause, ~3,400 pages of 10 items → ~115 min for the full catalog.

### Skinport — two distinct blocks, and they look alike

**1. HTTP 406, `Accept-Encoding`.** Skinport answers 406 to any request that does not advertise brotli, and `requests` advertises it only when a codec is importable. The historical "Cloudflare-dead" verdict was this, misdiagnosed. `brotli>=1.1.0` is pinned in `backend/requirements.txt` and `collectors/supply_depth.py::_probe_brotli` fails loudly rather than letting the feed disappear behind a header bug. macOS system `curl 8.7.1` is built without brotli, so it sends the header, gets a 200 and cannot decode the body.

**2. HTTP 403, egress ASN (new, 2026-08-06).** Separately, Skinport's WAF returns **403 with an HTML challenge page** to traffic from **Cloudflare-owned egress IPs (AS13335)**. No header change fixes it; measured with brotli decoding correctly and across every User-Agent tried. `scripts/probe_supply_feeds.py` reports this as a distinct `blocked_waf` status so it cannot be folded into (1) or into a fabricated zero.

Before concluding anything about Skinport, check `server: cloudflare` on the response and the caller's egress ASN — a 403 here is about *where you are calling from*. The coverage figures for Skinport `/v1/items` and `/v1/sales/history` in the two 2026-08-06 supply entries were measured from a residential IP and are **unverified from any other egress**.

`collectors/supply_depth.py` calls `/v1/items` daily and it contributed 0 rows on 2026-08-06. Skinport data arriving via the CSGOTrader aggregator is unaffected and correct (`starting_at`, not `last_24h`).

### market.csgo.com `volume` is a listing count

The field name is a landmine: in this repo `volume` means completed-sale count, refuted at |r| < 0.002. Verified as live inventory on 2026-08-06 by `scripts/probe_supply_feeds.py` on four grounds — a heavy right tail (median 13, p99 813, max 17,028); Spearman 0.56 against Waxpeer's `count`; values exceeding CSFloat's genuine daily sale count for all 9 canary items by a multiple that widens as liquidity falls (1.4x Kilowatt Case → 15x Glock Fade, the inventory ≈ trade rate × dwell time signature); and a per-physical-listing full export on the site. See `../changelog/2026-08-06-supply-depth-collector.md`.

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

> **⚠️ Superseded on the availability figures (2026-08-06).** Everything in this section
> was measured against an 11,092,908-row archive and reports the volume series running to
> 2026-03-29. The archive now holds **20,756,038 rows**, and volume is **identically zero
> for every row since 2026-04-16 — 111 days, across every source**. The last non-zero day
> is 2026-04-15 for buff163/youpin/csfloat and 2026-03-29 for `aggregator_sync`. The
> historical claim below still holds (2013–2025 is ~100% populated, 5,542 items); the
> "2026 is partial" line does not. See `data-inventory.md` §6 for the current per-period
> table. **The predictive verdict below is unaffected** — trade volume remains 0pp.

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

**Verdict (2026-07-16) — superseded 2026-08-06 on the availability half:** a free, bulk trade-volume source already exists *inside the archive* — the Steam price-history backfill (`aggregator_sync`) — for 5,542 items. Paid sources (CSMarketCap $9.99/mo, SteamWebAPI €15/mo) would only extend coverage to more items and keep recent days fresh; they do **not** add predictive signal.

> **Correction (2026-08-06).** The claim that the only free bulk trade-volume source is the *historical* in-archive backfill is wrong. Skinport `/v1/sales/history` is free, unauthenticated, bulk (one 1.9 s call), and **current** — 24h/7d/30d/90d volume for 36,004 items, 23,533 of them in the ≥$1 cohort. Nothing paid in the table above is needed to keep recent days fresh. The **second** half of the verdict is unaffected: no volume source has been shown to add predictive signal, and the |r| < 0.002 audit above still stands. See `docs/changelog/2026-08-06-retroactive-supply-feeds.md`.

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
4. Appends to `prices-YYYY.parquet` using the same dedup logic as `append_to_parquet.py`
   (it also wrote `snapshots-YYYY.parquet`, retired 2026-08-06 — see
   `../changelog/2026-08-06-price-archive-compaction.md`)

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

### The gate is IP *class*, not just IP reputation — VPNs cannot help

Tested 2026-08-06 on three distinct egresses:

| Egress | ASN / type | Result |
|---|---|---|
| Residential home connection | residential | Served hydrated pages, then penalised after ~300 requests |
| `138.199.35.122` DataPacket, Los Angeles US | AS212238, datacenter | **Stripped shell on request #1** |
| `185.107.80.83` NForce, Breda NL | AS43350, datacenter | **Stripped shell on request #1** |

Both VPN exits were blocked on their *first ever* request, with no traffic
history behind them — so this is not our rate-limit penalty following us, it is
Steam pre-blocking commercial hosting ranges. Same reason the route cannot run
from GitHub Actions. **A residential IP is the only viable egress**; switching
VPN region does not help, because the filter tracks ASN class, not country. The
only untested alternative with a real chance is a cellular hotspot, since carrier
ranges are residential-classified.

The residential penalty is long-lived: last real traffic 20:28, and probes at
22:56, 00:10, 01:09, 01:34 and 01:47 all returned the identical 251,388-byte
shell — **>5 h with no decay whatsoever**.

### Currency: a non-US egress silently corrupts prices

Steam renders logged-out market pages in the **visitor's geo-IP currency**. The
`STEAM_FEE_MULTIPLIER = 1.1607` divisor was calibrated against USD, so collecting
from, say, a Netherlands exit would return EUR and write plausible-looking but
wrong prices — a silent corruption, not a visible failure. Before trusting any
run from a new egress, check `"wallet_currency"` in the page (**1 = USD, 3 = EUR**)
and re-verify the ratio against overlapping archive rows.

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
| CSFloat `/api/v1/history/<name>/graph` | Yes | Daily completed-sale avg + count from 2020-04, no auth (1,899 rows for AK Redline) | **Measured 2026-08-06 and declined.** Coverage is fine (98% of the deep cohort, median 1,058 days) but a cross-market basis feature is null at every horizon and *harmful* at 30d, and `avg_price` is float-composition noise at the median 4 sales/day (within-item CV 1.347). Budget is **500 requests/day**, so a catalogue backfill is ~52 days. See `../changelog/2026-08-06-csfloat-basis-refuted.md`. |
| cs2.sh | No | Steam daily back to 2013 | No free tier; history is the $200/mo Scale plan and only reaches back to 2025-12-24. |
| CSGOSKINS.GG | No | 36 markets, 38.5K items | €179/mo minimum, 90-day history. |
| SteamAnalyst | Partly | 30+ markets | Free tier is 100 req/day — unusable in bulk. |
| Wayback Machine (`prices.csgotrader.app`) | Yes | — | `steam.json` has **zero** captures; only 4 stray snapshots of older filenames. Not a time series. |
| `HilliamT/scm-price-history` | Yes | The classic `var line1=` scrape | Dead — Steam moved to SSR. The React blob documented above is its replacement. |

## Free non-price feeds — exact pull recipes (verified 2026-08-06)

All verified 200 from a residential IP on 2026-08-06. No key, no cookie, no
signup on any of them. `item_slug` == `market_hash_name` in the archive, so every
one of these joins to `price-archive/prices-*.parquet` directly on name.

```bash
# 1. Skinport sales history — 24h/7d/30d/90d volume, 36,004 items, 20.4 MB, 1.9 s
#    RETURNS 406 WITHOUT `Accept-Encoding: br`. macOS curl 8.7.1 has no brotli
#    (`curl -V` → zlib only) and the backend venv has no `brotli` module, so pipe
#    the raw bytes through node to decode:
curl -s -H 'Accept-Encoding: br' \
  'https://api.skinport.com/v1/sales/history?app_id=730&currency=USD' -o sph.br
node -e "require('fs').writeFileSync('sph.json',require('zlib').brotliDecompressSync(require('fs').readFileSync('sph.br')))"
#    Shape: [{market_hash_name, last_24_hours:{min,max,avg,median,volume}, last_7_days:{…},
#             last_30_days:{…}, last_90_days:{…}}, …]   prices nullable, volume 0 when absent
#    NOTE: 854 duplicate market_hash_name entries — dedup on join.

# 2. lis-skins full export — 2,297,323 individual listings, 25,660 items, 173 MB, 44 s
curl -s --compressed 'https://lis-skins.com/market_export_json/api_csgo_full.json' -o lis.json
#    Shape: {"items":[{id, name, price, created_at, item_float, item_paint_seed,
#                      stickers, unlock_at, …}, …]}
#    created_at 100% populated; item_float 55.3%; unlock_at only 8 of 2.3M (unusable).

# 3. market.csgo.com — live listing count, 27,402 items, 2.6 MB, 0.63 s
curl -s --compressed 'https://market.csgo.com/api/v2/prices/USD.json' -o tm.json
#    Shape: {"success":true,"time":…,"currency":"USD",
#            "items":[{market_hash_name, volume, price}, …]}   volume is a STRING.

# 4. Steam item_nameid map — 26,935 entries, 227 KB (unblocks itemordershistogram)
curl -s --compressed \
  'https://raw.githubusercontent.com/somespecialone/steam-item-name-ids/master/data/CS2/item_names.json'
#    Shape: {"<market_hash_name>": <item_nameid int>, …}

# 5. Item metadata — crate first_sale_date, collection, float caps, StatTrak/Souvenir
curl -s --compressed 'https://raw.githubusercontent.com/ByMykel/CSGO-API/main/public/api/en/crates.json'
curl -s --compressed 'https://raw.githubusercontent.com/ByMykel/CSGO-API/main/public/api/en/skins.json'
#    crates.json 8.2 MB, 481 crates, 261 with first_sale_date.
#    skins.json  5.5 MB, 2,126 skins: rarity, min_float, max_float, stattrak,
#                souvenir, crates[], collections[].

# 6. Steam event calendar — keyless. `count=500` is ONE PAGE, not the feed's depth:
#    pass `enddate` to walk back. 4 pages / ~2 s → 1,752 unique items to 2012-03-16.
curl -s 'https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/?appid=730&count=500&maxlength=1'
curl -s 'https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/?appid=730&count=500&enddate=1646179200'
#    Shape: {"appnews":{"newsitems":[{gid,title,url,author,contents,feedlabel,date,…}]}}
```

Before implementing any of these as a collector, read
`docs/research/lis-skins-snapshot-plan.md` — it carries the constraints
(never run from `backend/`, verify from a GitHub runner first, return row counts
for the zero-row guard, guard the `min_ask` anchor) and the open `created_at`
ambiguity that decides whether the age features mean anything.

## Quality gaps
- Wire `data_validation.py` checks into the pipeline — it is still dead code (only importers are `collectors/__init__.py:1` and `tests/test_data_validation.py:6`; `pipeline.py:160` validates `price > 0` only)
- Historical fallback still emits flat-line rows with `timestamp=now`; they are labelled and tracked, but nothing downstream *excludes* them yet
