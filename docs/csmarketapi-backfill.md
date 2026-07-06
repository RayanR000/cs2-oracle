# CSMarketAPI Multi-Market Price History Backfill

## Goal

Backfill daily sales history (OHLCV per market) for CS2 items across all major trading platforms, prioritized by item popularity (liquidity). The data enables trend analysis, price prediction, and market intelligence without relying on a single source like Steam alone.

## Constraints

| Constraint | Detail |
|---|---|---|
| CSMarketAPI free tier | 1,000 requests/month/key |
| Accounts available | 5 (skrup.chezz, breadandpoops, rrane2025, rayanrane, bobafett) |
| Total monthly budget | ~4,750 requests (950 safety threshold × 5) |
| Items in local catalog | 31,908 (from `market_catalog.db`) |
| CSMarketAPI catalog | 31,417 items |
| Overlap | 26,718 items (in both catalogs) |
| CSMarketAPI-only | 4,699 items (not in local DB, no listing data) |
| No batch endpoints | Every item requires 1 API call — confirmed via OpenAPI spec, Python SDK (`csmarketapi` v2.0.0), and dashboard JS bundle |

## Architecture

### Database Split

Two separate databases to allow independent refresh cycles:

| Database | Contents | Size | Refresh Cadence |
|---|---|---|---|
| `runtime/csmarketapi.db` | `items`, `sales_history`, `backfill_state` | 1.8 GB | Monthly (per-key quota cycle) |
| `runtime/csmarketapi_reference.db` | `markets`, `currency_rates`, `player_counts` | 692 KB | Any time via `--refresh-ref` |

### Data Flow

```
market_catalog.db ──→ build_priority_queue() ──→ [item_1, item_2, ... item_n]
                                                        │
                     CSMarketAPI ──→ fetch_sales_history(hash_name) ──→ sales_history table
                          │
                     Key rotation (3 keys, round-robin)
                          │
                     checkpoint: last_hash_name → backfill_state
```

### Priority Queue

Items from `market_catalog.db` sorted by `sell_listings DESC`, with CSMarketAPI-only items appended at the end (no listing data = lowest priority).

| Tier | Listings | Items | Strategy |
|---|---|---|---|
| 1000+ | ≥1,000 | 1,960 | Highest priority — fetch first |
| 100-999 | 100–999 | 5,754 | High priority |
| 10-99 | 10–99 | 13,392 | Medium priority |
| 1-9 | 1–9 | 10,802 | Low priority |
| unknown | N/A | 4,699 | Lowest — appended at end |

### API Endpoints Used

**Bulk (1 request each, stored in reference DB):**

| Endpoint | Data | Rows |
|---|---|---|
| `GET /v1/markets` | Supported markets (SKINBARON, CSFLOAT, DMARKET, etc.) | 12 |
| `GET /v1/currency_rates` | Exchange rates (USD, EUR, CNY, RUB, INR) | 5 |
| `GET /v1/player_counts/history` | CS2 player count history (2011–2026) | 10,470 |
| `GET /v1/items` | Full item catalog | 31,417 |

**Per-item (1 request/item, stored in backfill DB):**

| Endpoint | Data |
|---|---|
| `GET /v1/sales/history/aggregate?market_hash_name=...&currency=USD` | Daily OHLCV per market |

### Key Rotation Logic

- Each key tracked via `req_idx_N` counter in `backfill_state`
- Safety threshold: **950/1,000** (leaves 50-request buffer)
- `find_key()` starts from `active_key_idx` and cycles through all keys
- On **429**: key is marked exhausted, next key tried, item failed
- When all keys at threshold: backfill pauses with checkpoint preserved

```python
def find_key(conn):
    for offset in range(len(keys)):
        idx = (start + offset) % len(keys)
        if req_count(conn, idx) < 950:
            return idx
    return None
```

## Script: `backend/collectors/csmarketapi_backfill.py`

### Usage

```bash
# Start or resume backfill (auto-uses all keys)
python backend/collectors/csmarketapi_backfill.py

# Test with limited items
python backend/collectors/csmarketapi_backfill.py --limit 50

# Preview priority queue without API calls
python backend/collectors/csmarketapi_backfill.py --dry-run

# Show progress + quota + per-market breakdown
python backend/collectors/csmarketapi_backfill.py --stats

# Reset checkpoint (keep data, restart from beginning)
python backend/collectors/csmarketapi_backfill.py --reset

# Re-fetch reference data only (markets, currency_rates, player_counts)
python backend/collectors/csmarketapi_backfill.py --refresh-ref
```

### Resilience Features

| Feature | Implementation |
|---|---|
| **SIGINT/SIGTERM** | Graceful shutdown — finishes current item, commits data, checkpoints |
| **Double-interrupt protection** | Second interrupt forces immediate exit |
| **Crash recovery** | On resume, checks `SELECT COUNT(*) FROM sales_history WHERE market_hash_name = ?` — skips items already committed |
| **Per-item checkpoint** | `last_hash_name` updated after each successful item commit |
| **Key rotation on 429** | Failed item logged, next key tried immediately |
| **Retry with backoff** | Server errors (5xx) retried up to 3 times with exponential backoff (2s, 4s, 8s) |
| **Atomic per-item commit** | DELETE old + INSERT new in single transaction — partial writes impossible |
| **Logging** | Simultaneous stdout + file (`runtime/logs/csmarketapi_backfill_*.log`) |

### Logging Detail

Each item logged with:

```
[2,847/36,607] (  7%) Sticker | Twistzz (Glitter) | Paris 2023
       via rrane2025        quota: 50/950  rate:0.50it/s  ETA:1130m  [  619 listings]
       ✓ 1,548 rows  [STEAMCOMMUNITY:1059, MARKETCSGO:263, CSFLOAT:169, WHITEMARKET:36, SKINPORT:13 …+1]
       (2,847 done, 0 failed)
```

Fields: item counter, total, percentage, hash name, active key, remaining quota, throughput, estimated time remaining, Steam listing count, row count, per-market breakdown (top 5), cumulative stats.

## Environment Setup

**`.env` file:**
```env
CSMARKETAPI_KEY_1=csmarketapi_key_w2n3aji1x6r8bivmy6me
CSMARKETAPI_ACCOUNT_1=skrup.chezz
CSMARKETAPI_KEY_2=csmarketapi_key_5lnvkqph98d0l8y8jmnl
CSMARKETAPI_ACCOUNT_2=breadandpoops
CSMARKETAPI_KEY_3=csmarketapi_key_eu1ku8kj24o2rjfut4ge
CSMARKETAPI_ACCOUNT_3=rrane2025
CSMARKETAPI_KEY_4=csmarketapi_key_57hz2y5alc6w04bx6dmc
CSMARKETAPI_ACCOUNT_4=rayanrane
CSMARKETAPI_KEY_5=csmarketapi_key_wceniweh2k6a7j792poc
CSMARKETAPI_ACCOUNT_5=bobafett
```

Supports up to 5 keys (loops `range(1, 6)`). Config model in `backend/config.py` exposes `settings.csmarketapi_keys` as a list of `{account, key}` dicts.

## Execution Results

### First Session (Full Burn)

| Phase | Items | Key Usage | Duration |
|---|---|---|---|
| Reference + catalog | — | 4 req on key 0 | ~30s |
| Key 0 (skrup.chezz) | ~950 items | 950 req | ~32 min |
| Key 1 (breadandpoops) | ~950 items | 950 req | ~32 min |
| Key 2 (rrane2025) | ~946 items | 950 req | ~32 min |
| **Total** | **2,846 items** | **2,850 req** | **~96 min** |

### Second Session (Burn Remaining 50/Key)

Rolled back `req_idx_1` and `req_idx_2` to 900 in the DB to re-expose ~50 quota each.

| Phase | Items | Key Usage |
|---|---|---|
| Key 2 (rrane2025) | ~50 items | Hit 429, rotated |
| Key 1 (breadandpoops) | ~50 items | Hit 429, rotated |
| All keys exhausted | — | Clean stop at item #2,942 |

### Final Totals

| Metric | Value |
|---|---|
| Items completed | **4,940** |
| Price rows | **~12,000,000+** |
| Failed | 256 (429 burns — all keys forced to 1000/1000) |
| Months of price data | ~4,500 unique days |
| Markets with data | 7 (STEAMCOMMUNITY, CSFLOAT, MARKETCSGO, WHITEMARKET, SKINPORT, SKINBARON, CSDEALS) |
| Database size | **~2.5 GB** (`csmarketapi.db`) |
| Key 0 actual usage | 1000 (hit 429) |
| Key 1 actual usage | 1000 (hit 429) |
| Key 2 actual usage | 1000 (hit 429) |
| Key 3 actual usage | 1000 (hit 429) |
| Key 4 actual usage | 1000 (hit 429) |

### Key Verification

All 5 keys confirmed exhausted via 429 response:

```json
{"detail": "You have exceeded your monthly quota. Consider upgrading your plan."}
```

## Key Decisions & Rationale

| Decision | Rationale |
|---|---|
| **Sales history over listings** | Sales are ground truth (actual transactions). Listings are ask prices (noise). |
| **Daily resolution** | CSMarketAPI returns daily ~OHLC. Sufficient for trend analysis; no need for intraday. |
| **sell_listings as priority signal** | Best available popularity proxy from local DB. Items with more Steam listings are more liquid. |
| **950 threshold over 1,000** | Safety margin — prevents mid-item 429. Can be temporarily lifted to 1000 for burn sessions (set `KEY_SWITCH_THRESHOLD = 1000` in script, then restore). |
| **Separate reference DB** | Allows re-fetching currency rates (change daily) and player counts (change hourly) without touching backfill state. |
| **1s delay between requests** | Respectful rate limiting. No documented rate limit, but avoids triggering abuse detection. |
| **Per-item commit + skip check** | Crash-safe: if script dies mid-write, the data for that item is incomplete but the item won't be re-fetched (checked on resume). |

## Resume Next Month

```bash
# Just run it — picks up automatically
python backend/collectors/csmarketapi_backfill.py

# Or refresh reference data first
python backend/collectors/csmarketapi_backfill.py --refresh-ref
python backend/collectors/csmarketapi_backfill.py
```

Next item to process: `Sir Bloody Skullhead Darryl | The Professionals` (#4,941 of 36,607).

### Future Optimizations

- **cs2.sh batch endpoint**: POST with 100 items/request. $75/mo Developer plan. Would reduce ~37K requests to ~370 requests.
- **Add more CSMarketAPI keys**: Each additional key adds ~950 items/month. Add `CSMARKETAPI_KEY_N` / `CSMARKETAPI_ACCOUNT_N` to `.env` and `config.py` (bump `range(1, 6)` to `range(1, N+1)`).
- **Parallel fetching**: Currently 1 request at a time (1s delay). Could parallelize with multiple keys simultaneously.
- **Selective date range**: Pass `start`/`end` params to sales history to reduce response size for items with very long histories.

## File Reference

| File | Purpose |
|---|---|
| `backend/collectors/csmarketapi_backfill.py` | Main backfill script |
| `backend/config.py` | Settings model with `csmarketapi_keys` property |
| `.env` | API keys + account names |
| `runtime/csmarketapi.db` | Backfill database (items + sales_history + state) |
| `runtime/csmarketapi_reference.db` | Reference database (markets + currencies + player counts) |
| `runtime/logs/csmarketapi_backfill_*.log` | Run logs |
