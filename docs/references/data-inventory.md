# Data Inventory & Coverage

**Measured 2026-08-06** against the local `price-archive/` working copy, post-compaction.
Companion to `data-sources.md`, which catalogues *where data comes from*; this file
records *what is actually on disk and how much of the market it covers*.

Read the caveats before quoting any number:

- The local `price-archive/` is a plain directory that **lags** the canonical
  `RayanR000/cs2-oracle-data` repo (`../architecture/data.md`). Counts here are the local
  copy's.
- Taken **after** `compact_price_archive.py` ran locally
  (`../changelog/2026-08-06-price-archive-compaction.md`). The canonical repo is still
  uncompacted, so its file list and size differ.
- The archive's max day is **2026-08-04**, two days behind the measurement date. Every
  "last 7 days" window below is anchored to 08-04.
- The `target_items` / `is_backfilled` gate lives in Postgres and has **no mirror in the
  archive**. Cohort figures use archive-side proxies, labelled as such.
- The ≥$1 split uses a plain cross-source median, **not** production's outlier-voted
  consensus, so it will shift by a small number of items under the production rule.

---

## 1. What is on disk

### Price series — the core

| File group | Files | Rows | Size | Range | Columns |
|---|---:|---:|---:|---|---|
| `prices-YYYY.parquet` | 13 | 9,429,275 | 33 MB | 2013-08-14 → 2025-12-31 | `item_slug, day, mean_price, volume` |
| `prices-YYYY-MM.parquet` | 9 | 11,326,763 | 56 MB | 2026-01-01 → 2026-08-04 | + `source` |
| **Total** | **22** | **20,756,038** | **89 MB** | 2013-08-14 → 2026-08-04 | |

`prices-2026-03` and `prices-2026-04` additionally keep `min_price`/`max_price` — they are
the only files where those columns hold real intraday range. `median_price` and the
`snapshots-*.parquet` family no longer exist; both were bit-for-bit redundant.

Largest single file: `prices-2026-07` at 6,173,061 rows / 20.9 MB (the multi-source
cutover month).

### Reference and auxiliary

| File | Rows | Size | Range | Holds |
|---|---:|---:|---|---|
| `item-metadata.parquet` | 8,691 | 0.1 MB | — | `item_slug, rarity, rarity_rank, weapon_type`. **Rarity NULL on 4,296 of 8,691** |
| `player-counts-*.parquet` (16) | 4,635 | 0.2 MB | 2011-11-30 → 2026-07-16 | Daily concurrents. **The 2026 file has 1 row** — collector removed |
| `exchange-rates-2026.parquet` | 306 | — | 2026-07-11 → 07-17 | **7 days only** |
| `snapshot-tier-history-through-2026-07-08.csv.gz` | — | 1.7 MB | ≤ 2026-07-08 | gzipped CSV, not Parquet |
| `player-counts/*.csv` | 5 | 20 KB | 2026-07-16 → 07-20 | uningested |
| `raw/17mafo/*.json` | ~90 | **605 MB** | 2026-04-18 → 2026-07-08 | Raw Steam scrape, already ingested. Six times the size of the entire Parquet archive |

### `ops/` mirrors

| Table | Rows | Size | Date range | Mirror mtime |
|---|---:|---:|---|---|
| `item_forecasts` | 158,200 | 1.6 MB | 2025-12-01 → 2026-08-05 | 08-05 |
| `forecast_outcomes` | 104,642 | 2.2 MB | 2025-12-01 → 2026-07-19 (targets to 07-31) | 08-02 |
| `prediction_accuracy` | 84 | — | 2025-12-02 → 2026-08-02 | 08-02 |
| `collection_runs` | 194 | — | 2026-05-29 → 2026-08-06 | 08-06 |
| `supply_snapshots` | 35,037 | 0.5 MB | **2026-07-15 only** | 07-20 |
| `event_impacts_denorm` | 18,473 | 0.7 MB | 2013-12-19 → 2026-05-10 | 07-20 |
| `events` | 79 | — | 2013-12-19 → 2026-05-10 | 07-20 |
| `accuracy_alerts` | 13 | — | — | 07-20 |

The mirrors are stale at **different** dates. `forecast_outcomes` in particular predates
the 2026-08-06 prod deletion of 14,233 stale NULL-`base_price` rows, so its NULL count
overstates prod.

---

## 2. Item coverage vs. the CS2 catalogue

There is no canonical count of CS2 market items, but the independent bulk feeds converge
on **~40,000 distinct `market_hash_name`s**:

| Feed | Items |
|---|---:|
| csgotrader aggregator (in-archive) | 39,316 |
| Skinport `/v1/sales/history` | 36,004 |
| market.csgo.com | 27,402 |
| Steam `item_nameid` map | 26,935 |
| lis-skins export | 25,660 |

That is 2,126 base skins (`ByMykel/CSGO-API`) expanded across wear × StatTrak × Souvenir,
plus cases, stickers, agents, patches, graffiti and music kits.

| | Count | % of ~40k catalogue |
|---|---:|---:|
| Distinct `item_slug` ever in the archive | 41,725 | — |
| …less the ~3,145 slug-keyed duplicates | **~38,600** | **~96%** |
| Seen in the last 7 days (07-29 → 08-04) | 41,423 | ~96% |
| Seen on the latest day (08-04) | 41,317 | ~96% |
| **Items actually forecast** (08-05 run) | **8,691** | **22%** |
| Items with pre-2026 history | **5,542** | **14%** |

**Breadth is effectively solved; the served cohort is not.** The pipeline touches nearly
the whole catalogue every day but forecasts a fifth of it.

### Served-cohort proxies

No `target_items` table exists in the archive. Two proxies, both 100% fresh:

| Proxy | Count | Fresh in last 7d |
|---|---:|---:|
| Items with any pre-2026 history | 5,542 | 5,542 (100%) |
| Items in `item-metadata.parquet` = items forecast on 08-05 | 8,691 | 8,691 (100%) |

5,542 matches the recorded prod `is_backfilled` count exactly, and `item_forecasts` on
2025-12-01 covers exactly 5,542 items, then 8,691 from 2026-07-29 onward — so the served
cohort **grew 5,542 → 8,691** between those dates.

---

## 3. The depth cliff

Distinct days of observation per item:

| Bucket | All 41,725 items | Share | Pre-2026 cohort (5,542) |
|---|---:|---:|---:|
| ≥730 days | 3,933 | 9.4% | 3,933 |
| ≥365 days | 5,144 | 12.3% | 5,144 |
| ≥180 days | 5,536 | 13.3% | 5,536 |
| ≥90 days | 25,092 | 60.1% | 5,542 |
| ≥30 days | 33,566 | 80.4% | 5,542 |
| <30 days | 8,159 | 19.6% | 0 |
| **Median** | **128 days** | | **1,507 days** |

**Every item with ≥180 days of history is in the 5,542 pre-2026 cohort.** The other 36,183
were first seen in 2026 and are capped at ~136 days. There is no middle tier — the archive
is two disjoint populations, 5,542 items with twelve years and ~36k with four months.

This is the single most consequential shape in the dataset. It is why depth-only sources
are declined (`buff-price-history-archive` pushes rows/item 1,341 → 1,641 and shrinks
`target_items` 521 → 426) and why the Steam listing-page backfill is valued for
*lowering* rows/item.

---

## 4. Time coverage

| Metric | Value |
|---|---|
| Earliest day | 2013-08-14 |
| Latest day | 2026-08-04 |
| Calendar span | 4,739 days |
| Distinct days present | 4,735 |
| **Missing days** | **4** |

Missing: **2026-07-27, 07-30, 08-02, 08-03**. All four are recent; 2013-08-14 → 2026-07-26
is unbroken. Consistent with midnight-UTC cron drift, not collector failure.

| Year | Rows | Items | Days | Sources |
|---|---:|---:|---:|---:|
| 2013 | 11,007 | 316 | 140 | 0 (NULL) |
| 2014–2019 | 2,678,596 | 458 → 1,872 | 365/366 each | 0 |
| 2020–2024 | 5,288,268 | 2,096 → 4,341 | 365/366 each | 0 |
| 2025 | 1,807,005 | 5,542 | 365 | 0 |
| **2026** | **11,326,763** | **41,725** | **212** | **13** |

**9,429,275 rows (45%) are pre-2026 with `source IS NULL`** — single-series backfill, no
market attribution.

---

## 5. Source breakdown

| Source | Rows | Items | First | Last | Days present | Missing in span | Stale vs 08-04 |
|---|---:|---:|---|---|---:|---:|---:|
| *(NULL — pre-2026 backfill)* | 9,429,275 | 5,542 | 2013-08-14 | 2025-12-31 | 4,523 | — | — |
| `aggregator_buff163` | 1,479,996 | 39,232 | 2026-03-22 | 2026-08-04 | 46 | **90** | 0 |
| `aggregator_youpin` | 1,284,399 | 36,948 | 2026-03-22 | 2026-08-04 | 46 | **90** | 0 |
| `aggregator_csfloat` | 1,187,964 | 31,741 | 2026-03-22 | 2026-08-04 | 46 | **90** | 0 |
| `aggregator_sync` | 1,063,366 | 36,960 | 2026-01-01 | 2026-08-04 | 110 | **106** | 0 |
| `aggregator_csgotrader` | 706,829 | 39,316 | 2026-07-11 | 2026-08-04 | 21 | 4 | 0 |
| `aggregator_steam_90d` | 680,597 | 37,007 | 2026-07-11 | 2026-08-04 | 21 | 4 | 0 |
| `aggregator_steam_30d` | 625,511 | 34,162 | 2026-07-11 | 2026-08-04 | 21 | 4 | 0 |
| `aggregator_buff163_buy` | 570,732 | 32,948 | 2026-07-11 | 2026-08-04 | 21 | 4 | 0 |
| `aggregator_steam_7d` | 538,364 | 31,299 | 2026-07-11 | 2026-08-04 | 21 | 4 | 0 |
| `aggregator_skinport` | 522,978 | 29,502 | 2026-07-11 | 2026-08-04 | 21 | 4 | 0 |
| `aggregator_csmoney` | 476,771 | 29,212 | 2026-07-11 | 2026-08-04 | 21 | 4 | 0 |
| `aggregator_steam_17mafo` | 2,169,483 | 27,304 | 2026-04-16 | 2026-07-10 | 85 | 1 | **25 — DEAD** |
| `historical_fallback:aggregator_sync` | 19,773 | 4,879 | 2026-07-11 | 2026-07-16 | 6 | 0 | **19 — retired** |

Naming note: the label is `aggregator_sync`, **not** `aggregator_steam_sync`, and there is
an `aggregator_steam_17mafo` family that is not in `data-sources.md`'s source table.

**The multi-source era is 25 days deep, not 4.5 months.** Eight of the eleven live sources
start 2026-07-11. Only buff163/youpin/csfloat reach back to 2026-03-22, and they have 90
missing days inside that span — they ran 03-22 → 04-15 (the HF dataset merge), then nothing
until 07-11. `aggregator_sync` is the only continuous 2026 series, with 106 missing days.
**Anything requiring cross-source history longer than ~3.5 weeks does not exist.**

`aggregator_steam_17mafo` has been dead for 25 days with nothing flagging it.

---

## 6. Field availability — one column carries information

NULL / zero rates over the last 90 days (2026-05-07 → 08-04, 8,305,434 rows):

| Column | NULL % | Zero % |
|---|---:|---:|
| `mean_price` | 0.000 | 0.000 |
| `volume` | 0.000 | **100.000** |
| `min_price` | **100.000** | — |
| `max_price` | **100.000** | — |
| `source` | 0.000 | — |

### Volume is present historically and dead currently

| Period | Rows | `volume > 0` |
|---|---:|---:|
| 2013–2025 | 9,429,275 | 9,429,275 (100%) |
| 2026-01 | 164,709 | 164,709 (100%) |
| 2026-02 | 140,414 | 140,414 (100%) |
| 2026-03 | 934,391 | 885,635 (94.8%) |
| 2026-04 | 1,631,398 | 1,181,424 (72.4%) |
| **2026-05 onward** | **8,455,851** | **0 (0%)** |

Non-zero for the whole 2013–2025 backfill (5,542 items, 4,523 days) and through
2026-04-15, then **identically zero for every row since 2026-04-16 — 111 days**. Last
non-zero day by source: buff163/youpin/csfloat 2026-04-15, `aggregator_sync` 2026-03-29.
The column is still written, and still read by `backtest/price_resolution.py:249`.

This supersedes the "Volume Data Status" figures in `data-sources.md`, which were measured
on 2026-07-16 against an 11.09M-row archive and report the series running to 2026-03-29.

### Other fields

| Signal | Reality |
|---|---|
| Intraday range | 2,466,349 rows carry min/max, all in 2026-03-22 → 04-30; 1,308,897 with `min != max`. Nowhere else |
| Bid/ask | Only `aggregator_buff163_buy` vs `aggregator_buff163`. 30,377 paired items on 08-04, median ask/bid **1.283**. **25 days deep** |
| Supply / listing counts | `ops/supply_snapshots.parquet`: 35,037 rows, 35,037 items, **all dated 2026-07-15**, source `steam_burst`. `sell_listings` populated, `skinport_quantity` NULL on all rows. A single cross-section with no time dimension — unusable as a lagged feature |
| Player counts | Stalled 2026-07-16; the 2026 file has one row |
| FX rates | 7 days — while buff163 and youpin, the two largest item-coverage sources, are CNY-denominated |

---

## 7. The cohort inversion

Latest-day cross-source median of `mean_price`, fallback sources excluded:

| Cohort | ≥$1 | <$1 | Total | % ≥$1 |
|---|---:|---:|---:|---:|
| All items with recent data | 26,468 | 14,955 | 41,423 | **63.9%** |
| Pre-2026 cohort (5,542) | 1,050 | 4,492 | 5,542 | 18.9% |
| Forecast cohort (8,691) | 1,423 | 7,268 | 8,691 | **16.4%** |

The archive at large is 64% dollar-plus. The cohort actually forecast is **84%
sub-dollar**. This is the concrete shape of the offline-DA/penny-item problem, and it runs
backwards: the items with tradeable signal are largely the ones the gate excludes.

---

## 8. Label coverage

`ops/forecast_outcomes.parquet` (mirror as of 2026-08-02):

| Metric | Value |
|---|---:|
| Total rows | 104,642 |
| Distinct items | 11,084 |
| Horizons | 4 (3, 7, 14, 30) |
| `forecast_date` range | 2025-12-01 → 2026-07-19 |
| Resolved (`actual_price` non-NULL) | 104,642 (100%) |
| **Usable (`base_price` non-NULL)** | **60,737 (58.0%)** |

| Horizon | Rows | Items | Usable | `forecast_date` range |
|---|---:|---:|---:|---|
| 3 | 38,689 | 11,084 | 22,093 | 2025-12-01 → 2026-07-19 |
| 7 | 38,658 | 11,084 | 22,143 | 2025-12-01 → 2026-07-19 |
| 14 | 16,474 | 10,976 | 11,040 | 2025-12-01 → 2026-07-17 |
| 30 | 10,821 | 10,821 | 5,461 | **2025-12-01 only** |

Three structural facts:

1. **There are only two forecast months** — 2025-12 (all on 2025-12-01, a backdated batch)
   and 2026-07. Nothing for Jan–Jun 2026. The label set is two snapshots, not a series.
2. **Horizon 30 exists only for 2025-12-01.** There is no 30-day label anywhere in 2026;
   every h=30 metric rests on 5,461 rows from one backdated batch.
3. **11,084 outcome items vs 8,691 forecast today** — ~2,400 outcome items are for items
   no longer served, consistent with the phantom-item purge.

`item_forecasts` covers only **6 distinct dates**: 2025-12-01, 2026-07-17/18/19 (5,542
items each), 2026-07-29 and 2026-08-05 (8,691 each). `prediction_accuracy` holds 76 daily
evaluations (2026-07-17 → 08-02) plus 8 walkforward rows. `collection_runs`: 194 runs over
51 days, 184 completed / 10 failed, no failures since 2026-07-21.

---

## 9. Reproducing these numbers

From `backend/`, via
`venv/bin/python -c "import duckdb; c=duckdb.connect(); print(c.sql('''…''').df().to_string())"`,
with `P` = `read_parquet('<repo>/price-archive/prices-*.parquet', union_by_name=true)`.

Totals:

```sql
SELECT count(*) AS n_rows, count(DISTINCT item_slug) AS items,
       min(CAST(day AS DATE)) AS mind, max(CAST(day AS DATE)) AS maxd,
       count(DISTINCT CAST(day AS DATE)) AS n_days
FROM P
```

Missing days:

```sql
WITH d AS (SELECT DISTINCT CAST(day AS DATE) d FROM P),
     b AS (SELECT min(d) a, max(d) z FROM d),
     cal AS (SELECT UNNEST(generate_series((SELECT a FROM b),(SELECT z FROM b),
                                           INTERVAL 1 DAY))::DATE d)
SELECT cal.d FROM cal LEFT JOIN d USING(d) WHERE d.d IS NULL ORDER BY 1
```

History-length distribution:

```sql
WITH h AS (SELECT item_slug, count(DISTINCT CAST(day AS DATE)) AS d FROM P GROUP BY 1)
SELECT count(*) FILTER (WHERE d>=730) AS ge730, count(*) FILTER (WHERE d>=365) AS ge365,
       count(*) FILTER (WHERE d>=180) AS ge180, count(*) FILTER (WHERE d>=90)  AS ge90,
       count(*) FILTER (WHERE d>=30)  AS ge30,  count(*) FILTER (WHERE d<30)   AS lt30,
       count(*) AS total, median(d) AS med_days
FROM h
```

≥$1 cohort (plain median — see the caveat at the top):

```sql
WITH r AS (SELECT item_slug, CAST(day AS DATE) d, mean_price FROM P
           WHERE source IS NOT NULL AND source NOT LIKE 'historical_fallback:%'
             AND mean_price IS NOT NULL AND CAST(day AS DATE) >= DATE '2026-07-29'),
     last AS (SELECT item_slug, max(d) d FROM r GROUP BY 1),
     px AS (SELECT r.item_slug, median(r.mean_price) AS p
            FROM r JOIN last USING(item_slug, d) GROUP BY 1)
SELECT count(*) FILTER (WHERE p>=1) AS ge1, count(*) FILTER (WHERE p<1) AS lt1,
       count(*) AS total
FROM px
```

Per-source span and staleness:

```sql
WITH s AS (SELECT source, CAST(day AS DATE) d FROM P WHERE source IS NOT NULL GROUP BY 1,2),
     b AS (SELECT source, min(d) a, max(d) z, count(*) have FROM s GROUP BY 1)
SELECT source, a AS mind, z AS maxd, have AS days_present,
       (z-a)+1 AS span_days, ((z-a)+1)-have AS missing_days, DATE '2026-08-04'-z AS days_stale
FROM b ORDER BY days_stale DESC
```

Volume availability by year:

```sql
SELECT year(day) AS y, count(*) AS n_rows,
       count(*) FILTER (WHERE volume=0) AS vol_zero,
       count(*) FILTER (WHERE volume>0) AS vol_pos,
       count(DISTINCT item_slug) FILTER (WHERE volume>0) AS items_vol_pos
FROM P GROUP BY 1 ORDER BY 1
```

---

## Related

- `data-sources.md` — where each feed comes from, and the pull recipes
- `../architecture/data.md` — how the archive is stored and read
- `../changelog/2026-08-06-price-archive-compaction.md` — why `median_price` and
  `snapshots-*` are gone
- `../changelog/2026-08-06-data-acquisition-ranking.md` — what to do about the gaps above
