# Volume Data Source Research

> 🔴 **THE VERDICT BELOW IS INVERTED (2026-08-15). Volume is FEED-BLOCKED, NOT REFUTED.**
> `changelog/2026-08-15-volume-features-remeasured.md` re-ran the arm on the post-2026-08-08
> statistics fix and the post-2026-08-13 trainer fix and found a **real directional gain at every
> horizon, placebo-clean**: **+1.503pp** [+0.631, +2.499] at 3d, **+1.879pp** [+0.031, +3.952] at
> 7d, **+1.442pp** [+0.277, +2.538] at 14d, against placebos that are null at all three. Eleven of
> the thirteen volume features survive the >0.95 prune.
>
> The shelving still stands, but **only because the live feed is dead** — not because the features
> lack signal. Anyone reading this doc for "is volume worth it?" should read it as **yes, if the
> feed is repaired**, which is precisely what makes `count_in_24` in
> `2026-08-16-refutation-power-tiers-and-iflow-backfill.md` valuable rather than optional.
>
> ⚠️ Its citation `backend/scripts/merge_hf_dataset.py:99` **no longer resolves** — the script was
> deleted 2026-08-10 in `fd49d5d`, collateral in a cleanup commit about something else. The claim
> is still true and still binding: `ask_volume AS volume` is at line 99 of `fd49d5d^`, and the rows
> it wrote are permanently in the archive, so **never pool across 2026-03-22** stands. The
> surviving prose reference is `backend/collectors/sales_volume.py:13`.

> **The conclusion stands; almost everything supporting it does not (2026-08-09).**
> `docs/changelog/2026-08-06-volume-ab-and-harness-defects.md:68` already named `:27`
> and `:142` as carrying wrong reasoning; they are unchanged three days later. Volume
> features are still refuted — on measured grounds, not these.
>
> **The |r| < 0.002 threshold is a cohort artifact.** The same changelog reproduced
> this doc's exact **n = 4,472,754** and found `price_momentum_7d` — the model's core
> feature — reads **−0.00143** on that cohort, while `volume_zscore_30d` on the ≥$1
> subset reads **+0.0424** (`:49-63`). A bar that calls the model's own best feature
> noise cannot classify anything.
>
> **The audit sample conditions on a sale having occurred.** Verified against the
> archive 2026-08-09: of the 9,429,275 pre-2026 rows, **zero** carry `volume = 0` and
> zero are NULL. A no-sale day is an *absent row*, not a zero. So the measured |r| is
> conditional on a trade, and cannot support `:29`'s generalisation to all rows.
>
> ~~**A free bulk trade-volume source already exists inside the archive**~~ (`:127`,
> `:144`) — **dead.** The last day with `volume > 0` anywhere is **2026-04-15**;
> 2026-04-16 → 2026-08-07 is uniformly `volume = 0`, non-NULL. (The 327,840 NULLs on
> 2026-08-08 are that day's partial write, not a state.) Mechanically,
> `prices.csgotrader.app/latest/steam_volume.json` 301s to an S3 `NoSuchKey` — it died
> upstream, not in the collector
> (`docs/research/2026-08-09-model-and-data-research.md` §4c). Archive-side the cliff
> reads as source substitution: on 04-16 the volume-bearing feeds are replaced by
> `aggregator_steam_17mafo`, and buff163/youpin/csfloat keep writing daily but write
> zeros.
>
> **Post-2026-03-22 "volume" is a listing count, not a trade count.**
> `backend/scripts/merge_hf_dataset.py:99` maps `ask_volume AS volume`. The three
> prefixed sources `aggregator_buff163`/`aggregator_youpin`/`aggregator_csfloat` carry
> **1,967,619** volume-bearing rows spanning **2026-03-22 → 2026-04-15**, and those are
> listing counts labelled volume. This doc's span ends 2026-03-29 only because it
> filtered to `source IS NULL OR source = 'aggregator_sync'`; re-running its query
> without that filter silently pools two different quantities, and the trio *overlaps*
> `aggregator_sync` on 03-22 → 03-29 rather than handing off to it. **Never pool across
> 2026-03-22.**
>
> **`:29` and `:146` are backwards.** All 13 volume features plus
> `volume_price_conf_{1,7}d` are in `SHELVED_FEATURES`
> (`backend/models/forecaster.py:405-431`) *because* they are dead on 100% of served
> rows — the opposite of "LightGBM handles the missing volume gracefully" — and they
> fall outside `FEATURE_GROUP_ALLOWLIST` (`:353`) regardless. `volume_price_conf`,
> named at `:29` as volume's surviving value, is shelved.
>
> **The coverage numbers are stale.** Re-measured against the archive 2026-08-09:
>
> | | doc says | actual |
> |---|---|---|
> | Volume-bearing rows | 88.65% of 11,092,908 | **53.15% of 22,203,660** (11,801,457) |
> | Items with volume | 5,542 | **27,714** (of 41,795) |
> | Span end | 2026-03-29 | **2026-04-15** |
> | 2026 share | 24.3% | **18.57%** (Jan/Feb 100%, Mar 94.8%, Apr 72.4%, **May–Aug 0.00%**) |
>
> The **9,833,838** figure does reproduce exactly, spanning 2013-08-14 → 2026-03-29,
> under this doc's own source filter.
>
> Two smaller corrections. ~~`:56` "all currently exhausted **this month**"~~ implies a
> monthly reset; `docs/references/backfill.md:19` records the CSMarketAPI keys as "all
> burned, and **they do not reset**", so the "~158 items/day (5 keys)" row at `:135` is
> a dead capability. And `:84-86`'s "CSFloat is not a free bulk option" is true of
> `/api/v1/listings` but **wrong for the history endpoint**
> (`docs/changelog/2026-08-06-free-bulk-supply-depth-feeds-exist.md:76-78`).

**Last updated: 2026-07-16**

## Current State (corrected 2026-07-16)

Volume **is** in the Parquet archive, and it is **not** limited to a 90-day window.

- **Coverage:** 9,833,838 rows (**88.65%** of all 11,092,908 price rows) carry non-zero `volume`, spanning **2013-08-14 → 2026-03-29** across **5,542 unique items**.
- **Source label:** tagged **`aggregator_sync`** in the archive (legacy scripts/backfill DB call it `STEAMCOMMUNITY` — same data).
- **Origin:** Steam price-history backfill (`scripts/backfill_ssr_history.py` → `steamcommunity.com/market/pricehistory/`); merged via `append_to_parquet.py`.
- **Per-year:** 2013–2025 ~100% volume-populated; 2026 partial (24.3%).

| Time Period | Data | Status |
|---|---|---|
| 2013 → 2025 | `aggregator_sync` Steam backfill, full years | ✅ Non-zero volume |
| Jan–Mar 2026 | `aggregator_sync` | ✅ Partial (24% of 2026 rows) |
| Apr 2026+ (live aggregator) | csgotrader / skinport / etc. | ❌ `volume=0` |
| CSMarketAPI backfill DB | `csmarketapi.db` | ❌ Not present on disk |

The CSMarketAPI backfill database (`runtime/csmarketapi.db`) that would have OHLCV sales history for ~4,940 items across 7 markets does not exist on the current machine (it was in `.gitignore` and likely lived on a different runner or was cleaned up).

The supply scraper collects `sell_listings` (active listings) but this is **not** trade volume — it's supply-side only.

### Does volume improve predictions? — No (verified 2026-07-16)

Tested on the volume-rich window (2023–2025, 4.47M samples with `volume>0`): every volume feature correlates with next-day and 7-day forward returns at **|r| < 0.002** (noise). Volume also fails to predict move *magnitude*. The only real predictive signal is **price momentum** (`corr(return_7d, fwd_return_7d) = +0.0796`).

**Implication:** volume features will **not** improve forecast accuracy. Volume's value is **data quality / confidence** (the `detect_market_manipulation` filter and `volume_price_conf` liquidity weighting), not prediction. The forecaster's volume features are only "zeroed" for the ~34K items that lack backfill volume — and even where present they add no predictive lift.

## Requirements

- No Steam cookie/session management
- Bulk calling preferred but not required
- Free tier preferred
- Daily volume updates to keep features fresh
- Must cover backfilled items (at minimum the tracked ~5.5K items, ideally all Steam items)

## Sources Evaluated

### CSGOTrader (already used)
- Endpoints: `prices.csgotrader.app/latest/steam.json` etc.
- Data: `last_24h`, `last_7d`, `last_30d`, `last_90h` — **prices only, no volume**
- Free, bulk (all items in one endpoint)
- Already integrated; cannot add volume from this source

### CS2Cap
- **Free** ($0/mo): `GET /prices/candles` — OHLCV with `v` (estimated trade volume) and `q` (active listings). 1K req/month, 20 req/min. Per-item.
- **Starter** ($19/mo): `POST /prices/batch` — listing count (`quantity`) for 100 items/req. 50K req/month.
- **Pro** ($79/mo): `GET /market/items` — ALL items in one call, returns `sales_1d/7d/30d`, `total_volume_24h`, `supply`, `liquidity`.
- No cookies needed. API key auth.

### CSMarketAPI (csmarketapi.com — already used for backfill)
- **Free** ($0/mo): 1K req/month. `GET /v1/sales/latest/aggregate` returns `volume` (sales count). Per-item (requires `market_hash_name`).
- **Pro** ($9.99/mo): 1M req/month.
- 5 keys available (all currently exhausted this month at 1,000/1,000 each)
- Not bulk — per-item, so even 5 keys × 950 safe req = ~158 items/day

### Skinstrack
- **Free**: 50 req/month — too few for production.
- **Paid** ($24.99/mo): Volume data across 34+ marketplaces.

### SteamWebAPI
- **Item Small** (€15/mo ≈ $16.50): `GET /steam/api/items` — **ALL items in one call**. Returns `sold24h`, `sold7d`, `sold30d`, `offervolume`, `buyordervolume`. Best value option for paid.
- Free plan cannot access the bulk items endpoint.

### cs2.sh
- **Developer** ($75/mo): Prices only.
- **Scale** ($200/mo): Includes `/v1/archive/csfloat` (daily completed-sale volume since 2022). Too expensive.

### Steam Community Market (priceoverview endpoint — undocumented)
- `GET https://steamcommunity.com/market/priceoverview/?appid=730&market_hash_name=...`
- Returns `volume` (items sold in last 24h), `lowest_price`, `median_price`
- **No auth required** (no cookies, no API key)
- **Per-item** (not bulk), rate-limited to ~10-15 req/min
- Risk of IP ban under sustained load
- At 10 req/min: ~600 items/hour, ~9 hours for 5.5K items
- Free but too slow for daily full-catalog coverage

### Steam Community Market (pricehistory endpoint)
- Returns per-item price history with volume.
- **Requires valid Steam session cookie** — user does not want to manage cookies.

### CSFloat (direct API)
- Public listing endpoint returns 403 without API key.
- API requires key for read access; not a free bulk option.

### CSMarketCap API (csmarketcap.com — distinct from CSMarketAPI)
- **Standard** ($9.99/mo): 10K requests/month, 30 req/min
- **Pro** ($49.99/mo): 50K requests/month, 60 req/min
- `POST /api/v2/rest/get-steam-analytics` — **returns ALL items in one call**
- Returns per-item: `sales { last_24h, last_7d, last_30d, last_90d, avg_daily_volume }`, plus listing and buy order stats
- JWT token auth (expires 2h, auto-refreshable via SDK)
- Website features free; API requires paid subscription
- 1 call/day = 30/month — well within 10K quota
- Cheapest paid option discovered

### SkinWise
- Free, no auth
- `GET /api/v1/price/{slug}` — per-item
- Only prices, **no volume data**

### CSPriceAPI
- Free tier with sensible rate limits (unclear exact limits)
- Per-item pricing, unknown if volume included
- Not thoroughly evaluated

### PriceEmpire (pricempire.com)
- **Free testing tier** (~100 test calls)
- **Hobby** ($30/mo): 1K credits/month, 20 req/min
- **Developer** ($100/mo): 5K credits/month, 100 req/min
- `get_skin_price_history` returns `[timestamp, price, volume]` tuples
- Per-item (1 credit = 1 API call)
- Free tier too limited; paid tiers expensive
- Has unofficial Parse API wrapper (free 100 credits/mo)

### cs2.sh (re-evaluated)
- Free: **2-day developer key only** (not viable for ongoing use)
- Developer ($75/mo): unlimited requests, 10 req/s
- `GET /v1/prices/latest` — truly bulk (all items in 1 call)
- Returns `ask_volume`/`bid_volume` (listing counts, **not trade volume**)
- For actual trade volume: `POST /v1/archive/steam` (daily purchase vol since 2013) — Scale plan ($200/mo)
- Does not provide daily trade volume data on the Developer plan

## Verdict (corrected 2026-07-16)

A free, bulk trade-volume source **already exists inside the archive** — the Steam price-history backfill (`aggregator_sync`) — covering 5,542 items from 2013–2026. Every *external* option that has reliable, fresh trade volume still charges for API access, but that is now a **coverage/freshness** question, not a "we have no volume at all" problem.

| Source | Cost | Bulk? | Trade Volume Fields | Daily Coverage for 5.5K items |
|--------|:----:|:-----:|:-------------------|:------------------------------|
| **Steam price-history backfill (already in archive)** | Free | n/a (historical) | daily trade volume | 5,542 items, 2013–2026 |
| Steam `priceoverview` | Free | ❌ per-item | 24h sales count | ~600 items/hr (too slow) |
| CSMarketCap API | **$9.99/mo** | ✅ 1 call | `last_24h/7d/30d/90d` | ✅ All items, 1 call |
| SteamWebAPI Item Small | €15/mo | ✅ 1 call | `sold24h/7d/30d/90d` | ✅ All items, 1 call |
| CSMarketAPI (your keys) | Free | ❌ per-item | sales_history.volume | ~158 items/day (5 keys) |
| CS2Cap Pro | $79/mo | ✅ batch 1K | `sales_1d/7d/30d` | ✅ 1 batch call |
| Pricempire Standard | $99.90/mo | ❌ per-item | trade count metas | ❌ free tier too small |
| cs2.sh Developer | $75/mo | ✅ bulk | ask_volume (listings, not trade vol) | ❌ no trade vol on dev plan |

## Recommendation

**Volume sourcing is not justified by prediction accuracy.** The data shows volume features add no predictive lift (|r| < 0.002 with forward returns). Before paying for any volume API:

1. **Use what's already in the archive.** The `aggregator_sync` Steam backfill already provides free, multi-year trade volume for 5,542 items.
2. **Only consider a paid source (CSMarketCap $9.99/mo / SteamWebAPI €15/mo) if the goal is broader *coverage* (the ~34K items without backfill) or *fresh* daily volume for liquidity/confidence weighting — not better forecasts.**
3. The earlier "volume zeros degrade 7d/14d features" concern is moot: those features are driven by *price*, and volume's role is data-quality, not prediction. LightGBM handles the missing volume gracefully.
