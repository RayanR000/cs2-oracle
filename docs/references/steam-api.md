# Steam Market API Reference

Findings from testing — do not re-test, use this as reference.

> **The rate-limit envelope below applies to residential IPs only.**
>
> **Hosted GitHub Actions runners are 429'd on the very first request.** There is no interval, burst pattern, or backoff that makes this work from CI — the runner IP ranges are blocked outright. This is why `supply-scraper.yml` was deleted (commit 0288568) and why `supply_snapshots` is frozen at 35,037 rows.
>
> So: "safe interval 3 seconds", "full catalog in ~2.85 hours", "5s delay is safe" are all measurements from a residential connection. Any plan that schedules Steam scraping in GitHub Actions is dead on arrival; it needs a residential/proxied egress or a local run.

---

## Endpoints

### 1. `/market/search/render/` — Item Catalog

**URL:** `https://steamcommunity.com/market/search/render/`

**Purpose:** List all items on the CS2 Steam Market with current snapshot data.

**Auth:** None required (public endpoint).

**Consumers:** the catalog builder (`scripts/build_market_catalog.py`, `scripts/repair_catalog_gaps.py`) and `collectors/supply_scraper.py`. There is **no working discovery entry point** — `scripts/discover_steam_items.py:20` does `from collectors.real_data_collector import get_collector` and no `real_data_collector.py` exists, so `discover-new-items.yml` dies with an ImportError before any Steam call. That workflow is **manual-dispatch only** (its schedule was disabled 2026-07-08), so the breakage only surfaces on a one-off run — but item onboarding via this endpoint is unreachable either way.

**Parameters:**

| Param | Value | Notes |
|---|---|---|
| `appid` | `730` | CS2 app ID |
| `norender` | `1` | Returns raw JSON |
| `start` | `0, 10, 20, ...` | Offset (page number × 10) |
| `count` | `100` | **Ignored — always returns 10 items.** `collectors/supply_scraper.py:48,158,173` still sends `count=100` while advancing `start` by 10 — a 10× over-fetch, still open. |
| `category_730_Type[]` | `tag_CSGO_Tool_Sticker` | Only works for Stickers (returns 15,349). All other categories return 0. |
| `q` | search string | **Does not filter** — always returns all 34,263 items |

**Response structure:**

```json
{
  "success": true,
  "start": 0,
  "pagesize": 10,
  "total_count": 34263,
  "searchdata": { ... },
  "results": [
    {
      "name": "Dreams & Nightmares Case",
      "hash_name": "Dreams & Nightmares Case",
      "sell_listings": 419569,
      "sell_price": 176,
      "sell_price_text": "$1.76",
      "sale_price_text": "$1.69",
      "app_icon": "https://...",
      "app_name": "Counter-Strike 2",
      "asset_description": {
        "appid": 730,
        "classid": "4717330486",
        "background_color": "393b3e",
        "icon_url": "i0CoZ81Ui0m-...",
        "tradable": 1,
        "name": "Dreams & Nightmares Case",
        "name_color": "b0c3d9",
        "type": "Base Grade Container",
        "market_name": "Dreams & Nightmares Case",
        "market_hash_name": "Dreams & Nightmares Case",
        "commodity": 1,
        "market_bucket_group_name": "Dreams & Nightmares Case",
        "market_bucket_group_id": "G18D2253004"
      }
    }
  ]
}
```

**Data fields:**

| Field | Type | Description |
|---|---|---|
| `name` | string | Display name |
| `hash_name` | string | Unique market identifier — use this for `pricehistory` API |
| `sell_listings` | int | Current active sell listings count |
| `sell_price` | int | Current lowest price in **cents** (176 = $1.76) |
| `sell_price_text` | string | Formatted price string |
| `sale_price_text` | string | Sale price if discounted |
| `asset_description.type` | string | Item category (see type list below) |
| `asset_description.tradable` | int | 1 = tradeable, 0 = not |
| `asset_description.commodity` | int | 1 = commodity (stackable), 0 = unique |
| `asset_description.classid` | string | Asset class ID |
| `asset_description.name_color` | string | Hex color for UI display |
| `asset_description.icon_url` | string | Icon path (prepend `https://shared.fastly.steamstatic.com/community_assets/images/`) |
| `asset_description.market_bucket_group_id` | string | Bucket grouping ID |

**Item types observed (first 5000 items):**

| Type | Count | Category |
|---|---|---|
| Mil-Spec Grade Rifle | many | Skin |
| Classified Rifle | many | Skin |
| Restricted Rifle | many | Skin |
| Covert Rifle | many | Skin |
| StatTrak™ variants | many | Skin (with StatTrak) |
| Souvenir variants | many | Skin (Souvenir) |
| ★ Covert Knife | many | Knife |
| Base Grade Container | ~15 | Container/Case |
| Exotic Sticker | many | Sticker |
| High Grade Sticker | many | Sticker |
| Remarkable Sticker | many | Sticker |
| Superior Agent | many | Agent |
| Distinguished Agent | many | Agent |
| High Grade Music Kit | many | Music Kit |
| High Grade Collectible | many | Collectible/Pin |
| Restricted Equipment | few | Equipment (Zeus) |
| High Grade Charm | few | Charm |
| Base Grade Tool | few | Tool (StatTrak Swap) |

---

### 2. `/market/pricehistory/` — Historical Price Data

**URL:** `https://steamcommunity.com/market/pricehistory/`

**Purpose:** Full price history for a single item (time series).

**Auth:** Required — session cookies (`sessionid` + `steamLoginSecure`).

**Parameters:**

| Param | Value |
|---|---|
| `appid` | `730` |
| `market_hash_name` | item's `hash_name` from search/render |

**Response:**

```json
{
  "success": true,
  "prices": [
    ["Jul 02 2014 01: +0", 39.268, "1112"],
    ["Jul 03 2014 01: +0", 38.5, "980"],
    ...
  ]
}
```

Each record: `[date_str, price_float, volume_string]`

**Rate limits:** 12-15 req/min before 429. 5s delay between requests is safe.

---

## Rate Limits

**All figures in this section were measured from a residential IP.** From a hosted GitHub runner the first request already returns 429 — see the note at the top of this file.

### search/render

| Metric | Value |
|---|---|
| Hard page size | 10 items (cannot increase) |
| Burst limit | ~10-12 rapid requests before 429 |
| Recovery after 429 | ~30 seconds |
| Safe interval | 3 seconds between requests |
| Items per request | 10 |
| Total pages for 34,263 items | 3,427 |
| Estimated time (3s interval) | ~2.85 hours |

### pricehistory

| Metric | Value |
|---|---|
| Safe interval | 5 seconds between requests |
| Burst limit | ~12-15 req/min before 429 |
| Recovery after 429 | ~30-60 seconds |
| Total items | ~34,263 (after catalog build) |
| Estimated time (5s interval) | ~47 hours |

### Ban behavior

- 429 = temporary rate limit (30s recovery)
- Sustained 429s = IP ban (hours, renewing if you keep hitting)
- **Datacenter IPs are not "banned faster" — they are pre-banned.** Hosted CI runners 429 on request #1 with no prior traffic.
- Session cookie expiry = all requests return empty (hard to distinguish from items with no history)

---

## Market Totals

**Snapshot from the catalog build (2026-07-04); not re-measured since.** `catalog-build.md`
records the total as ~34,301 from the same build, so treat 34,263 as ±40 and re-read
`total_count` before relying on it.

| Metric | Value |
|---|---|
| Total items on CS2 market | 34,263 |
| Stickers | 15,349 |
| Skins (weapons) | ~18,000 |
| Containers/Cases | ~100 |
| Agents | ~100 |
| Music Kits | ~50 |
| Collectibles/Pins | ~50 |
| Equipment (Zeus) | few |
| Charms | few |
| Tools | few |

---

## Catalog Coverage

The item catalog is **`backend/runtime/market_catalog.db`** — 18 MB, **31,908 `market_items`** (re-counted 2026-08-21; file unchanged since 2026-07-04), ~93% of the ~34,263 items on the market. See `catalog-build.md` for the per-category breakdown and how it was built.

The catalog is maintained by the backfill/catalog-build scripts, **not** by Steam discovery: `discover-new-items.yml` is both unscheduled (since 2026-07-08) and broken at import (see above), so nothing adds items from `search/render/` on a schedule. Any older "Production DB vs Market" table (24,822 items / 5,712 stickers / ~72% coverage) is stale — that was the pre-catalog Supabase item list.
