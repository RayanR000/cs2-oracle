# The 30-day wait is superseded — two supply feeds ship their own history

**Date:** 2026-08-06
**Change:** no code change. Research only. This extends
`2026-08-06-free-bulk-supply-depth-feeds-exist.md`, which refuted the
*availability* premise behind `2026-07-16-drop-supply-depth.md`. That entry
closed with a caveat that is now wrong for two of the feeds. The **accuracy**
claim from 07-16 is still untouched by everything below.

## The caveat this record retires

From `2026-08-06-free-bulk-supply-depth-feeds-exist.md:91-92`:

> **The 30-day wait is real.** All three feeds are live-only, no history.
> Nothing is testable until ~2026-09-05 at the earliest.

That was true of the three feeds audited that morning (Skinport `/v1/items`,
Waxpeer `/v1/prices`, Bitskins `/market/insell/730`) — each returns a scalar
count for *now*, so `supply_change_7d` and `supply_listings_zscore` needed 30
days of accumulation before they existed at all.

Two feeds found later the same day carry a **retroactive window**, so the
velocity/turnover shape is computable on the first pull rather than after 30
days of collection. Whether it is worth computing is a separate and still
unanswered question — see "What has not changed".

All figures below were measured live from a residential IP on 2026-08-06.
Coverage is a DuckDB join against `price-archive/prices-2026-08.parquet`:
**41,381** distinct `item_slug`, of which **26,428** are in the ≥$1 cohort
(`median(median_price) >= 1`, grouped by `item_slug`). `item_slug` equals
`market_hash_name` in the archive, so the joins are direct on the name.

Note the ≥$1 denominator here (26,428) is not the 25,459 used in the morning
entry's coverage table — the cohort was computed with a different aggregation.
**Percentages in this record are not directly comparable to that table**; the
raw matched counts are.

## Skinport `/v1/sales/history` — 90 days of trade history, one call, 1.9 s

```
GET https://api.skinport.com/v1/sales/history?app_id=730&currency=USD
```

Unauthenticated. **36,004 items**, 20,369,367 bytes decompressed, **1.9 s**.
Per item it returns `min`, `max`, `avg`, `median` and `volume` for each of
`last_24_hours`, `last_7_days`, `last_30_days`, `last_90_days`.

| Cut | Items in archive | ≥$1 cohort | % of ≥$1 cohort |
|---|---:|---:|---:|
| Any match | 34,895 | 23,533 | **89.0** |
| `volume > 0`, 90d | 22,433 | 14,799 | 56.0 |
| `volume > 0`, 30d | 16,242 | 11,110 | 42.0 |
| `volume > 0`, 7d | 9,774 | 6,814 | 25.8 |

The 89.0% is name-match coverage; the volume rows are the ones that actually
carry a number. Four nested windows on one pull is not a daily series, but it
is enough to compute a 7d-vs-30d and 30d-vs-90d turnover ratio per item on day
one.

**Trap: HTTP 406 without `Accept-Encoding: br`.** This is the third recording of
the same trap (`docs/references/data-sources.md:12, 42`). Worse, the macOS
system `curl 8.7.1` is built **without** brotli — `curl -V` lists only `zlib` —
so it sends the header, gets a 200, and cannot decode the body. The backend venv
does not have `brotli` either. Decode with node's
`zlib.brotliDecompressSync` or a Python environment where `brotli` is installed.

## lis-skins full export — 2.3M individual listings with creation timestamps

```
GET https://lis-skins.com/market_export_json/api_csgo_full.json
```

Unauthenticated. **2,297,323 listings** across **25,660** distinct items,
172,882,643 bytes, **44 s**. Shape is `{"items": [...]}`; per-listing keys:
`id, name, price, unlock_at, item_class_id, created_at, item_asset_id, game_id,
is_internal, item_link, item_float, name_tag, item_paint_index, item_paint_seed,
stickers`.

- `created_at` populated on **100.0%** of listings; `item_float` on 55.3%.
- Coverage: 25,160 archive items; **17,453** of the ≥$1 cohort (**66.0%**).
- 212,136 listings were created within the prior 24 h. Oldest live listing:
  **2026-03-15**.

This is qualitatively different from the three scalar-count feeds and from
market.csgo.com below. It is the full **ask ladder** plus **listing age**, which
makes computable on the first pull:

- depth within *x*% of the lowest ask (shape, not just count),
- median time-on-market per item, from `created_at` on live listings,
- supply inflow rate — 212,136 listings/day, per item.

The 44 s / 173 MB pull is the largest of anything audited; a collector would want
to reduce to per-item aggregates rather than store the ladder.

## market.csgo.com — a fourth depth feed the morning audit missed

```
GET https://market.csgo.com/api/v2/prices/USD.json
```

Unauthenticated. **27,402 items**, 2,581,089 bytes, **0.63 s**. Per item:
`market_hash_name`, `volume` (listing count), `price`. 27,159 items with
`volume > 0` matched the archive; **17,274** of the ≥$1 cohort (**65.4%**).

Live-only, like the original three — it does not carry history and does not
retire the 30-day caveat. It is recorded here because it belongs in the same
table and was absent from the morning audit.

## Non-market sources verified the same day

- **`somespecialone/steam-item-name-ids`** (GitHub, last pushed 2026-08-03, 90
  stars). `data/CS2/item_names.json`, 226,706 bytes, **26,935**
  `market_hash_name` → `item_nameid` entries. The morning entry listed
  `steamcommunity.com/market/itemordershistogram` as "blocked on per-item
  `item_nameid` mapping and Steam's rate limits". This removes **the first half
  only**. The endpoint is Steam-hosted, so it inherits the IP-class finding in
  `docs/references/data-sources.md:203-219` — residential egress only, no GitHub
  runner, no datacenter VPN — and it is per-item, not bulk. The rate limit was
  and remains the binding constraint.
- **`ByMykel/CSGO-API`** (raw GitHub JSON, free). `crates.json`: 481 crates, 261
  carrying `first_sale_date`. `skins.json`: 2,126 skins with rarity,
  `min_float`/`max_float`, `stattrak`, `souvenir`, and crate/collection
  membership. `price-archive/item-metadata.parquet` currently carries only
  `item_slug`, `rarity`, `rarity_rank`, `weapon_type` — so **item age, crate
  cohort and collection membership are absent today**. This is cross-sectional
  metadata, not a time series; it cannot produce a velocity feature.
- **Steam news.** `api.steampowered.com/ISteamNews/GetNewsForApp/v2/?appid=730&count=500`
  — no key, **0.29 s**, 500 entries back to **2022-03-01**. A free CS2 event
  calendar. Relevant to `event_correlation_analysis.py`, which per the
  event-correlation finding reads a Postgres table that is empty by design.

## Negative results — directions now closed

**The Wayback Machine holds nothing.** CDX queried with `matchType=prefix`,
day-collapsed, returned **zero captures** for all six of:

- `api.skinport.com/v1/items*`
- `api.skinport.com/v1/sales/history*`
- `api.waxpeer.com/v1/prices*`
- `market.csgo.com/api/v2/prices/USD.json*`
- `api.bitskins.com/market/insell/730*`
- `prices.csgotrader.app/latest/*`

There is **no retroactive listing-count history recoverable from the internet
archive**. Skinport's 90-day windows are the only substitute. This also
independently reconfirms the `prices.csgotrader.app` result already in
`data-sources.md:260`.

**Dead or gated on probe:**

| Endpoint | Result |
|---|---|
| shadowpay `/api/v2/user/items/prices` | `{"status":"error","error_message":"need_auth"}` |
| haloskins | 404 |
| gamerpay `/items` | 400 |
| skinsback `/api/market/prices` | 404 |
| c5game | 302 |
| csgobackpack.net | 301 (Cloudflare, dead) |

**Answer, but carry strictly less than lis-skins:**

| Endpoint | Result |
|---|---|
| `cs.deals/API/IPricing/GetLowestPrices/v1?appid=730` | 200, 26 KB, lowest price only, no depth |
| `skinout.gg/api/market/items?pagination=false&float_min=0` | 200, 856,885 bytes, 6.5 s, 20,197 rows with `total_count` and float |

## What has NOT changed

Nothing in this record measures directional lift. Every reason to be sceptical
of supply depth from 07-16 and from this morning survives intact:

- **Trade volume correlates with forward returns at |r| < 0.002 across 4.47M
  rows** (`docs/research/volume-data.md`). That audit stands. Skinport's
  `volume` fields are trade volume and fall squarely under it.
- **The listing-count *level* is a liquidity signal at ~0pp.** Only the
  change/velocity variant was ever argued predictive, at a calibrated
  **+1–2pp**.
- **`FEATURE_GROUP_ALLOWLIST = ["price_technicals"]`** in
  `backend/models/forecaster.py` exists because 85 non-price features measured
  within fold noise (7-fold purge-gap CV, 2026-07-24).
- **The genuinely new quantity is listing age and ask-ladder shape** from
  lis-skins. It is mechanistically distinct from the trade-volume level that was
  refuted — a distribution over time-on-market and distance-from-lowest, not a
  count. It is **untested, not refuted.** No lift is claimed or implied here.
- **The test design is unchanged and should not be relitigated.** Run
  `backend/scripts/compute_mde.py` first, then a **permutation** A/B, never a
  plain A/B. If the MDE comes back above ~2pp, none of this is measurable and
  the collector should not be built.
- **Unverified from a GitHub runner.** All three feeds are non-Steam hosts, so
  the 429 that killed `supply-scraper.yml` should not apply — but that is an
  inference, not a measurement. It is the first thing to check before building
  a collector.

## Docs touched

- `docs/references/data-sources.md` — added rows for Skinport
  `/v1/sales/history`, lis-skins, market.csgo.com, the `item_nameid` map,
  `ByMykel/CSGO-API` and Steam news to the source table, and marked the
  "Volume data sources evaluated (2026-07-16)" verdict as superseded: Skinport
  `/v1/sales/history` is a free, bulk trade-volume source that is *also*
  current, which that verdict said did not exist.

The four docs listed as carrying the now-false availability claim in
`2026-08-06-free-bulk-supply-depth-feeds-exist.md:132-143` are still uncorrected
apart from `data-sources.md`.

## Still open

- Whether any of this is measurable at all — blocked on `compute_mde.py`.
- Runner-IP behaviour for all four depth feeds.
- lis-skins reduction strategy: 173 MB/day of raw ladder is not something to
  archive as-is.
- Backfilling `item-metadata.parquet` with crate/collection/age from
  `ByMykel/CSGO-API`, which is independent of the supply question.

## Related

- `docs/changelog/2026-08-06-free-bulk-supply-depth-feeds-exist.md` — the entry
  this extends
- `docs/changelog/2026-07-16-drop-supply-depth.md` — the original decision; its
  accuracy leg still stands
- `docs/changelog/2026-08-06-volume-features-shelved.md`
- `docs/research/volume-data.md` — the trade-volume audit that still stands
- `docs/references/data-sources.md:203-219` — the IP-class finding that bounds
  the `itemordershistogram` route
