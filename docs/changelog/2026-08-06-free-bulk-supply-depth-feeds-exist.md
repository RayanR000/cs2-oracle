# "No free bulk listing-count source exists" is false — three of them do

**Date:** 2026-08-06
**Change:** no code change. One of the three rationales behind
`2026-07-16-drop-supply-depth.md` — the **availability** claim — is refuted by
measurement. The **accuracy** claim from that decision is untouched and still
stands. Supply depth moves from *dropped on cost* to *untested, and now cheap
enough to test*.

## What the 07-16 decision said

> - **Free source is too slow; no free bulk source exists.** The only free path is
>   the Steam full-catalog burst scrape (~115 min/day) … Steam `priceoverview` is
>   per-item (~8h for 5.5K) and does not return listing count.
>   **No free bulk listing-count API exists.**

That was the load-bearing premise. The same doc had already concluded that the
*change/velocity* variant is the mechanistically predictive one and the *level*
is not — so the reason the predictive variant could not be built was that
`supply_change_7d` and `supply_listings_zscore` need ≥30 days of daily
listing-count history, and there was no affordable way to accumulate it.

## What was found

Three marketplace endpoints return a full per-item listing count in **one
unauthenticated call**. Measured from a residential IP on 2026-08-06:

| Feed | Endpoint | Items | Depth field | Other non-price fields | Pull |
|---|---|---|---|---|---|
| Skinport | `api.skinport.com/v1/items?app_id=730&currency=USD` | 25,143 | `quantity` (Σ 3,644,106) | `min_price`, `max_price`, `mean_price`, `median_price`, `created_at`, `updated_at` | 10.3 MB / **1.9 s** |
| Waxpeer | `api.waxpeer.com/v1/prices?game=csgo` | 22,882 | `count` (Σ 1,380,250) | `type`, `rarity_color`, `steam_price` | 4.9 MB / **2.5 s** |
| Bitskins | `api.bitskins.com/market/insell/730` | 10,920 | `quantity` (Σ 492,375) | `price_min`, `price_max`, `price_avg` | 1.4 MB / **0.2 s** |

No key, no cookie, no signup on any of the three. Skinport **requires**
`Accept-Encoding: br` — `gzip` returns HTTP 406, which is the same trap recorded
in `data-sources.md` for the historical "Cloudflare-dead" misdiagnosis.

Coverage against the live universe (41,381 distinct `item_slug` in
`price-archive/prices-2026-08.parquet`):

| Feed | Items in archive | % of archive | Items in ≥$1 cohort | % of ≥$1 cohort |
|---|---:|---:|---:|---:|
| Bitskins | 10,894 | 26.3 | 5,122 | 20.1 |
| Waxpeer | 22,535 | 54.5 | 16,081 | 63.2 |
| Skinport | 24,878 | 60.1 | 15,875 | 62.4 |
| **Union** | **27,966** | **67.6** | **18,311** | **71.9** |

The ≥$1 cohort (25,459 items) is the one production reports on, so **71.9%** is
the number that matters, not 67.6%. Total cost of a daily snapshot of all three:
**~5 s and ~17 MB.**

These are not Steam hosts, so the runner-IP 429 that killed `supply_scraper.yml`
should not apply — **untested from a GitHub runner**, and that is the one thing
worth checking before building anything.

## The historical leg, which 07-16 also assumed was paid-only

`csfloat.com/api/v1/history/<market_hash_name>/graph` needs no key and returns
daily **completed-sale count** plus average price:

| Item | Days | Range | Latency |
|---|---:|---|---:|
| `AK-47 \| Redline (Field-Tested)` | 1,899 | 2020-04-02 → 2026-08-06 | 0.34 s |
| `Kilowatt Case` | 899 | 2024-02-07 → 2026-08-06 | 0.12 s |

10 sequential requests returned 10× HTTP 200 in 2.7 s (**3.7 req/s**). Headers
carry `x-ratelimit-limit: 500` on a sliding window; `x-ratelimit-remaining`
bounced between 145 and 498 across calls, so the budget is per-backend and the
effective ceiling was not established — throttling, not bandwidth, governs a
full-catalog backfill.

Note this is **trade volume**, not listing count, and therefore falls under the
|r| < 0.002 audit — its value here is that it is the only free path to
*historical* per-item volume that is also current, which is exactly the shape of
the train/serve gap that shelved thirteen features this morning
(`2026-08-06-volume-features-shelved.md`). `volume-data.md:84-86` records CSFloat
as "not a free bulk option"; that is correct for `/api/v1/listings` and wrong for
this endpoint.

## What has *not* changed

**The accuracy claim stands.** Nothing here measures directional lift. Still true
from 07-16, and still the reason to be sceptical:

- Trade volume correlates with forward returns at |r| < 0.002 (4.47M rows).
- The listing-count *level* is a liquidity signal, ~0pp. Only the change/velocity
  variant was ever argued to be predictive, at a calibrated **+1-2pp**.
- `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` exists because the 85
  non-price features measured within fold noise (2026-07-24, 7-fold purge-gap CV).

**The 30-day wait is real.** All three feeds are live-only, no history. Nothing
is testable until ~2026-09-05 at the earliest.

**The test design is already specified** by 07-16 and should not be relitigated:
*permutation* A/B on the change/velocity features only, never a plain A/B, per
the player-count capacity-inflation lesson. Run `scripts/compute_mde.py` first —
a +1-2pp target against the harness's known noise floor is the whole question,
and if the MDE comes back above 2pp the 30 days of collection buys nothing
measurable.

## Incidental findings

- **The aggregator can never supply this.** All seven `prices.csgotrader.app`
  dumps were re-inspected: `steam`, `skinport`, `buff163`, `csfloat`, `csmoney`,
  `youpin`, `csgotrader`. None carries a quantity or listing-count field. The
  supply side has to come from a new collector or not at all.
- **The bid side is already collected.** `buff163.json` carries `highest_order`
  alongside `starting_at`, and `csgotrader_aggregator.py:330-334` already stores
  it as `aggregator_buff163_buy` — 31,233 items in 2026-08. Bid-ask spread is
  computable today from data already in the archive, with no new source.
- **`price-archive/player-counts-*.parquet` is unused.** 4,925 daily rows,
  2011-11-30 → 2026-07-16, with mean/peak/min/reading_count. Nothing under
  `backend/` reads it; the only references are the dead CSMarketAPI backfill. It
  is stale because that quota died, but
  `api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/?appid=730`
  needs no key (returned 559,054). Note 07-16's warning that player counts are
  the original capacity-inflation cautionary tale.
- **`steamcommunity.com/market/itemordershistogram`** returned a full
  `sell_order_table` **and** `buy_order_table` unauthenticated from this IP.
  Deepest microstructure signal available free; blocked on per-item
  `item_nameid` mapping and Steam's rate limits.
- **Steam `priceoverview`** also answered unauthenticated (`Kilowatt Case` →
  `volume: 124,482`). Per-item, no listing count — 07-16's description of it is
  accurate.
- **Dead or gated, checked and ruled out:** DMarket v1 retired (HTTP 410) and v2
  `/marketplace-api/v2/offers` requires auth (401); SkinBaron `ProductStatistics`
  404; CS.MONEY `api.cs.money/1.0/market/sell-orders` connection failure;
  Tradeit `/v2/inventory/data` 404; CSGOEmpire IP/country-blocks. cs2.sh has
  full-depth Steam orderbooks since 2026-06-09 but its free tier is a **2-day**
  developer key.

## Docs carrying the now-false claim

Not yet edited — these need the correction applied:

- `docs/changelog/2026-07-16-drop-supply-depth.md:22-25` — the premise itself.
- `docs/research/accuracy-opportunities.md:71` — the 🛑 DECISION callout repeats
  "no free bulk listing-count source exists" as rationale (3).
- `docs/research/volume-data.md:84-86` — CSFloat "not a free bulk option",
  true only of the listings endpoint.
- `docs/references/data-sources.md` — the source table has no row for any of the
  three feeds, and the "Volume data sources evaluated (2026-07-16)" table lists
  only paid options above the free-archive verdict.

## Related

- `docs/changelog/2026-07-16-drop-supply-depth.md` — the decision this corrects
- `docs/changelog/2026-08-06-volume-features-shelved.md` — the train/serve gap
  the CSFloat leg would address
- `docs/changelog/2026-07-15-supply-scraper.md`,
  `docs/changelog/2026-07-15-supply-side-features.md`
- `docs/research/volume-data.md` — the trade-volume audit that still stands
