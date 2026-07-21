# Data Collection Pipelines

Four independent pipelines feed the Parquet archive + Supabase:

| Pipeline | Frequency | Source | Data Written | Workflow |
|----------|-----------|--------|-------------|----------|
| CSGoTrader Mult-Market | 23:00 UTC daily | `prices.csgotrader.app/latest/` (7 markets) | `prices-YYYY.parquet`, `snapshots-YYYY.parquet`, `collection_runs` | `aggregator-update.yml` |
| Supply Scraper | 22:00 UTC daily | Steam Community Market (pagination) | `supply_snapshots` (sell_listings) | `supply-scraper.yml` |
| Social Sentiment | 05/11/17/23 UTC (6-hourly) | old.reddit.com (3 CS2 subreddits) | `social_mentions` | `reddit-sentiment.yml` |
| HF Dataset Merge | One-time | HuggingFace CS2 hourly dataset (69.2M rows) | `prices-2026.parquet`, `snapshots-2026.parquet` | Manual |

---

## CSGoTrader Multi-Market Aggregator

### Data Sources (from `prices.csgotrader.app/latest/`)
| Endpoint   | Source Label (DB/Parquet)    | Fields Used                              |
|------------|------------------------------|------------------------------------------|
| steam.json | `aggregator_sync`            | `last_24h`, `last_7d`, `last_30d`, `last_90d` |
| skinport.json | `aggregator_skinport`     | `starting_at`                            |
| buff163.json  | `aggregator_buff163`      | `starting_at.price`                      |
| buff163.json  | `aggregator_buff163_buy`  | `highest_order.price`                    |
| csfloat.json  | `aggregator_csfloat`      | `price`                                  |
| csmoney.json  | `aggregator_csmoney`      | `price`                                  |
| csgotrader.json | `aggregator_csgotrader` | `price`                                  |
| youpin.json    | `aggregator_youpin`      | `price`                                  |

### Files
- **`collectors/csgotrader_aggregator.py`** — Fetches all 4 endpoints in one session; returns `Dict[str, SourceData]` per item with `{source: (price, volume, timestamp)}`. Logs failed endpoint counts, match rates, and warns on low match rate.
- **`collectors/pipeline.py`** — Maps sources to DB labels; writes all sources to snapshot CSV for Parquet archive (no prices written to Supabase — only `CollectionRun` records). Logs `"ZERO items collected"` error when nothing is returned.
- **`scripts/append_to_parquet.py`** — Accepts `--snapshot-csv` for all-source flat data and `--backfilled-csv` (legacy). Writes `prices-YYYY.parquet` (OHLCV) and `snapshots-YYYY.parquet` (all sources). Warns on missing/empty CSV files.
- **`scripts/run_task.py`** — Exits with code 1 when `items_collected == 0`, triggering GitHub failure notification.
- **`.github/workflows/aggregator-update.yml`** — Daily 23:00 UTC schedule.

### Storage Strategy
- **Supabase**: Only `CollectionRun` tracking records — all price data is CSV → Parquet only. (Prices in Supabase are stale.)
- **Parquet** (`archive/price-archive/`):
  - `prices-{YYYY}.parquet` — Steam daily OHLCV (from `aggregator_sync` rows)
  - `snapshots-{YYYY}.parquet` — Flat rows of all sources (`item_slug`, `day`, `source`, `price`, `volume`)

### Coverage Per Run
- ~18K `aggregator_sync` rows + ~30K multi-market rows = ~48K total/day (~1.2 MB/day)

---

## Supply Scraper

### What it does
Paginates the Steam Community Market (34K-item catalog) to collect sell_listings (supply) counts for each item. Uses burst rate limiting: 20 rapid requests → 30s pause. Full run takes ~115 min.

### Files
- **`collectors/supply_scraper.py`** — Pagination logic, rate limiting, HTML parsing.
- **`scripts/run_supply_scraper.py`** — Standalone entry point.
- **`.github/workflows/supply-scraper.yml`** — Daily 22:00 UTC, 120-min timeout.

### Storage
- `supply_snapshots` table in Supabase — daily sell_listings per item (~11K rows/day).
- Supply depth features (`supply_listings_log`, `supply_zscore_30d`, `supply_change_7d`, `supply_to_volume_ratio`) feed the forecaster.

---

## Social Sentiment Collector

### What it does
Scrapes Reddit for CS2 skin mentions every 6 hours. Monitors 3 subreddits (`GlobalOffensiveTrade`, `csgomarketforum`, `CSGOSkinInvesting`) via old.reddit.com HTML (Reddit's JSON API was killed in May 2026). Regex-matches skin names from ~2000 item names, scores each mention using VADER sentiment.

### Files
- **`collectors/social_sentiment.py`** — Reddit scraper + VADER scoring.
- **`db/parquet.py`** — Dual-write to Parquet ops archive + Supabase.
- **`.github/workflows/reddit-sentiment.yml`** — 6-hourly at 05/11/17/23 UTC, 10-min timeout.

### Known Limitation
VADER is a 2014 general-purpose lexicon — CS2 market jargon ("BFK CW MW low float") scores as neutral. After 3 days of production data, none of the 5 social features rank in top 20 by gain importance. Recommendation: replace with ModernFinBERT (ONNX INT8).

### Features Written
`social_mentions_1d`, `social_mentions_7d`, `social_mention_velocity`, `social_sentiment_7d`, `social_score_7d` — dual-written to `social_mentions` (Supabase) and `ops/social_mentions.parquet`.

---

## HF Dataset Merge (One-Time)

### What it did
Merged the Hugging Face "CS2 Market Data" dataset (CC BY 4.0, 69.2M hourly rows, 32K items, Mar 22 – Apr 15 2026 from BUFF/CSFloat/YouPin) into the Parquet archive. Filled the 17-day gap (Mar 30–Apr 15) and expanded coverage of 8 overlap days (Mar 22–29) by ~32K items.

### Files
- **`scripts/merge_hf_dataset.py`** — Downloads, deduplicates, merges HF data into Parquet.

### Impact
- `prices-2026.parquet`: 19 MB → 44.6 MB, 2.0M → 4.1M rows
- `snapshots-2026.parquet`: 7.6 MB → 21.2 MB, 1.6M → 3.7M rows
- Remaining gap: Apr 16 – Jul 8 (84 days) still unfilled.

---

## Test Coverage
- **134 tests** passing across 6 test files — aggregator fuzzy matching, pipeline + DB flow, forecaster ML, regime-switching, data validation, fallback recovery.
