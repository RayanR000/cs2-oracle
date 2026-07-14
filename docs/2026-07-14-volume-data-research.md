# Volume Data Source Research

## Current State

The Parquet archive has historical trade volume (Jan–Jun 2026) from the Steam backfill. July 2026 has all zeros — the daily aggregator (`CSGOTraderAggregator`) only collects prices, **not** volume. The forecaster already has volume features (`volume_lag_1d`, `volume_mean_7d`, `volume_zscore_30d`, etc.), but they receive zero for recent days, degrading 7d and 14d window features.

## Requirements

- No Steam cookie/session management
- Bulk calling preferred but not required
- Free tier preferred
- Daily volume updates to keep features fresh

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

### CSMarketAPI
- **Free** ($0/mo): 1K req/month. `GET /v1/sales/latest/aggregate` returns `volume` (sales count). Per-item (requires `market_hash_name`).
- **Pro** ($9.99/mo): 1M req/month.

### Skinstrack
- **Free**: 50 req/month — too few for production.
- **Paid** ($24.99/mo): Volume data across 34+ marketplaces.

### SteamWebAPI
- **Item Small** (€15/mo ≈ $16.50): `GET /steam/api/items` — **ALL items in one call**. Returns `sold24h`, `sold7d`, `sold30d`, `offervolume`, `buyordervolume`. Best value option for paid.
- Free plan cannot access the bulk items endpoint.

### cs2.sh
- **Developer** ($75/mo): Prices only.
- **Scale** ($200/mo): Includes `/v1/archive/csfloat` (daily completed-sale volume since 2022). Too expensive.

### Steam Community Market (pricehistory endpoint)
- Returns per-item price history with volume.
- **Requires valid Steam session cookie** — user does not want to manage cookies.

### CSFloat (direct API)
- Public listing endpoint returns 403 without API key.
- API requires key for read access; not a free bulk option.

## Verdict

For a **free** solution, neither CS2Cap (1K req/mo) nor CSMarketAPI (1K req/mo) can cover all 5,542 items daily — only ~33 items/day. They are viable only for strategically targeting the most volatile/traded items.

For **paid**, SteamWebAPI Item Small (€15/mo) is the best option:
- Cheaper than CS2Cap Starter ($19/mo)
- **Single call** returns all items (true bulk)
- Returns actual trade volume (`sold24h/7d/30d`), not just listing count
- Also includes price data (could simplify pipeline)

## Recommendation (Deferred)

The highest-ROI improvement without paying for volume data is implementing **cross-item correlation features** (cases ↔ collection skins). This uses existing price data and likely provides a bigger accuracy lift than volume data. The volume gap (~10 days of zeros) mainly affects 7d/14d features; 30d/60d features still have good signal from Jan–Jun data, and LightGBM handles the NaN gracefully.
