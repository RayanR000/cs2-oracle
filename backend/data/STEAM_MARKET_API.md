# Steam Market API Reference

Findings from testing — do not re-test, use this as reference.

---

## Endpoints

### 1. `/market/search/render/` — Item Catalog

**URL:** `https://steamcommunity.com/market/search/render/`

**Purpose:** List all items on the CS2 Steam Market with current snapshot data.

**Auth:** None required (public endpoint).

**Parameters:**

| Param | Value | Notes |
|---|---|---|
| `appid` | `730` | CS2 app ID |
| `norender` | `1` | Returns raw JSON |
| `start` | `0, 10, 20, ...` | Offset (page number × 10) |
| `count` | `100` | **Ignored — always returns 10 items** |
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
- Datacenter IPs banned faster than residential
- Session cookie expiry = all requests return empty (hard to distinguish from items with no history)

---

## Market Totals

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

## Production DB vs Market

| | Production DB | Steam Market | Coverage |
|---|---|---|---|
| Skins | 24,737 | ~18,900 | ~100% (includes StatTrak/Souvenir) |
| Cases | 85 | ~85 | ~100% |
| Stickers | 5,712 | 15,349 | ~37% |
| Agents | 0 | ~100 | 0% |
| Music Kits | 0 | ~50 | 0% |
| Collectibles | 0 | ~50 | 0% |
| **Total** | **24,822** | **34,263** | **~72%** |

Missing: ~9,440 items (mostly stickers, agents, music kits, collectibles, charms).
