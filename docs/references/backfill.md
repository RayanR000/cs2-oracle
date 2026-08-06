# CSMarketAPI Multi-Market Price History Backfill

> **DEAD — the data described here is not on disk and cannot be re-fetched.**
>
> 1. `backend/runtime/csmarketapi.db` is **0 bytes**. Every "Final Totals" figure below (4,940 items, ~12M rows, 7 markets, ~2.5 GB) describes data that no longer exists locally. Treat this document as a design record, not an inventory.
> 2. The free-tier quota is **permanently exhausted** — every key still returns 429 a month after the burn, so the monthly reset the plan assumed never happens.
> 3. With an empty DB there is also no `backfill_state` checkpoint, so a resume would restart from item #1 even if quota existed.
>
> The only surviving CSMarketAPI artifact is `backend/runtime/csmarketapi_reference.db` (692 KB: markets, currency rates, player counts). Historical CSMarketAPI *prices* that reached the Parquet archive are still there — see `data-sources.md` — but the source DB is gone.

## Goal

Backfill daily sales history (OHLCV per market) for CS2 items across all major trading platforms, prioritized by item popularity (liquidity). The data enables trend analysis, price prediction, and market intelligence without relying on a single source like Steam alone.

## Constraints

| Constraint | Detail |
|---|---|---|
| CSMarketAPI free tier | 1,000 requests/month/key — **all burned, and they do not reset** |
| Accounts available | 5 burned (skrup.chezz, breadandpoops, rrane2025, rayanrane, bobafett); code supports **6** slots (`config.py:49-50,56` → `range(1, 7)`) |
| Total monthly budget | Nominally ~4,750 requests (950 safety threshold × 5) — **actually 0** |
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
| `runtime/csmarketapi.db` | `items`, `sales_history`, `backfill_state` | **0 bytes — empty** (was ~2.5 GB) | Never (quota dead) |
| `runtime/csmarketapi_reference.db` | `markets`, `currency_rates`, `player_counts` | 692 KB | Any time via `--refresh-ref` |

### Data Flow

```
market_catalog.db ──→ build_priority_queue() ──→ [item_1, item_2, ... item_n]
                                                        │
                     CSMarketAPI ──→ fetch_sales_history(hash_name) ──→ sales_history table
                          │
                     Key rotation (up to 6 keys, round-robin)
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

**Open bug — 429s get no backoff.** `csmarketapi_backfill.py:205-207` logs the 429 and `return None` immediately, inside the same retry loop where 5xx sleeps `2 ** attempt` and continues (:212-214). A transient/burst 429 is therefore indistinguishable from a quota kill and costs the item outright. This is the mechanism behind the 256 "failed" items below.
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
CSMARKETAPI_KEY_1=your_key_here
CSMARKETAPI_ACCOUNT_1=account_1
CSMARKETAPI_KEY_2=your_key_2_here
CSMARKETAPI_ACCOUNT_2=account_2
CSMARKETAPI_KEY_3=your_key_3_here
CSMARKETAPI_ACCOUNT_3=account_3
CSMARKETAPI_KEY_4=your_key_4_here
CSMARKETAPI_ACCOUNT_4=account_4
CSMARKETAPI_KEY_5=your_key_5_here
CSMARKETAPI_ACCOUNT_5=account_5
CSMARKETAPI_KEY_6=your_key_6_here
CSMARKETAPI_ACCOUNT_6=account_6
```

Supports up to **6** keys (`config.py:49-50,56` declares slots 1–6 and loops `range(1, 7)`). Config model in `backend/config.py` exposes `settings.csmarketapi_keys` as a list of `{account, key}` dicts. All keys are `Optional[str] = None`, so unset slots are simply skipped.

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

### Final Totals (historical — none of this is on disk)

These were the totals when the burn finished. `csmarketapi.db` is now 0 bytes, so read this as a record of what the quota bought, not as available data.

| Metric | Value |
|---|---|
| Items completed | **4,940** |
| Price rows | **~12,000,000+** |
| Failed | 256 (429 burns — all keys forced to 1000/1000) |
| Months of price data | ~4,500 unique days |
| Markets with data | 7 (STEAMCOMMUNITY, CSFLOAT, MARKETCSGO, WHITEMARKET, SKINPORT, SKINBARON, CSDEALS) |
| Database size at the time | **~2.5 GB** (`csmarketapi.db` — now 0 bytes) |
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

## Resume Status: not resumable

The original plan — "wait for the monthly quota reset, rerun, it picks up from the checkpoint" — is dead on **both** legs:

1. **Quota never resets.** All keys still return `{"detail": "You have exceeded your monthly quota. Consider upgrading your plan."}` more than a month after the burn. The free tier's "1,000 requests/month" does not behave as a rolling monthly allowance for these accounts.
2. **No checkpoint survives.** `csmarketapi.db` is 0 bytes, so `backfill_state` (and its `last_hash_name` / `req_idx_N` counters) is gone. Even with working quota, a rerun would restart the priority queue at item #1 and re-spend requests on items already fetched once.

Restarting this capability requires a paid plan (or a different bulk source) **and** an accepted full re-fetch from scratch. Do not describe it as a resume.

### Coverage by Tier (as of the burn — historical)

| Tier | Items in Queue | Fetched | Remaining | Coverage |
|------|:-------------:|:-------:|:---------:|:--------:|
| **1000+** (most liquid) | 1,960 | 1,824 | 136 | **93.1%** |
| 100-999 | 5,754 | 2,794 | 2,960 | 48.6% |
| 10-99 | 13,392 | 0 | 13,392 | 0% |
| 1-9 | 10,802 | 0 | 10,802 | 0% |
| CSAPI-only (0 listings) | 4,699 | 0 | 4,699 | 0% |

The 136 remaining high-priority items (1000+ tier) include capsules, stickers, and skins not yet reached. No items were skipped due to queue ordering — all popular items in the local DB were correctly prioritized. The `build_queue` function was hardened to resolve CSMarketAPI-only items from the `items` table (defensive improvement, no queue order change).

### Future Optimizations

- **cs2.sh batch endpoint**: POST with 100 items/request. $75/mo Developer plan. Would reduce ~37K requests to ~370 requests.
- **Add more CSMarketAPI keys**: superseded — the range was already bumped to `range(1, 7)` (6 slots), and new free accounts do not help because the exhausted keys never recover. Adding slots beyond 6 still means editing `config.py` and `.env` together.
- **Fix the 429 backoff** (`csmarketapi_backfill.py:205-207`): give 429 the same exponential retry as 5xx so a burst limit doesn't discard the item.
- **Parallel fetching**: Currently 1 request at a time (1s delay). Could parallelize with multiple keys simultaneously.
- **Selective date range**: Pass `start`/`end` params to sales history to reduce response size for items with very long histories.

## Complementary Coverage: Hugging Face Dataset (2026-07-20)

The CSMarketAPI backfill reached **4,940 items** (Steam + 6 markets) before the quota died. A complementary expansion — which, unlike the backfill DB, *is* still in the Parquet archive — was done via the [HF CS2 Historical Item Price Dataset](https://huggingface.co/datasets/idomanteu/cs2-historical-item-prices-hourly-march-april-2026) (CC BY 4.0):

| Metric | CSMarketAPI Backfill | HF Dataset |
|--------|:--------------------:|:----------:|
| Items | 4,940 | **32,848** |
| Markets | 7 (Steam + 6) | 3 (BUFF, CSFloat, YouPin) |
| Period | 2013–2026 (daily) | Mar 22 – Apr 15 2026 (hourly) |
| Data type | Sale prices (OHLCV) | Ask/bid OHLCV |
| Method | Per-item API calls | Bulk Parquet download |
| Auth | API keys ($9.99/mo) | CC BY 4.0 (free) |

The HF dataset brought the total items with some historical data from **5,542 to ~33,000** for the Mar 22 – Apr 15 window. See `docs/references/data-sources.md` for full details. Script: `backend/scripts/merge_hf_dataset.py`.

## File Reference

| File | Purpose |
|---|---|
| `backend/collectors/csmarketapi_backfill.py` | Main backfill script |
| `backend/scripts/merge_hf_dataset.py` | HF dataset merge into Parquet archive |
| `backend/config.py` | Settings model with `csmarketapi_keys` property |
| `.env` | API keys + account names |
| `runtime/csmarketapi.db` | Backfill database (items + sales_history + state) — **0 bytes / empty** |
| `runtime/csmarketapi_reference.db` | Reference database (markets + currencies + player counts) |
| `runtime/logs/csmarketapi_backfill_*.log` | Run logs |
