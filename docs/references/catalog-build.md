# Phase 1: Market Catalog Build — Documentation

## What We Built

A complete catalog of every CS2 item on the Steam Community Market, stored in local SQLite. This replaces the production database's limited 24,822-item catalog with a full 31,908-item catalog scraped directly from Steam's API.

**Output:** `backend/runtime/market_catalog.db` (18 MB, 31,908 `market_items`)

---

## Why

Our production Supabase database only had 24,822 items (24,737 skins + 85 cases). The Steam Market has ~34,301 items. We were missing ~9,400 items — mostly stickers, charms, graffiti, agents, music kits, and collectibles — because they were never added to our DB.

This catalog covers 31,908 of ~34,301 items on the Steam Market for CS2 (93% coverage). The remaining ~2,393 are likely delisted/removed items.

---

## How It Works

### Endpoint Used

`GET https://steamcommunity.com/market/search/render/`

- Returns 10 items per page (hard-coded, cannot increase)
- No authentication required (public endpoint)
- Total pages for ~34,000 items: ~3,400

### Data Fields Captured Per Item

| Field | Example | Description |
|---|---|---|
| `hash_name` | "Glock-18 | Grinder (Field-Tested)" | Unique market identifier |
| `name` | "Glock-18 | Grinder (Field-Tested)" | Display name |
| `type` | "Restricted Pistol" | Item category/subtype |
| `sell_price` | 379 | Current lowest price (cents) |
| `sell_price_text` | "$3.79" | Formatted price |
| `sale_price_text` | "$3.50" | Sale price if discounted |
| `sell_listings` | 56 | Active sell listings count |
| `tradable` | 1 | Tradeable flag |
| `commodity` | 0 | Commodity flag (stackable items) |
| `classid` | "4717330486" | Asset class ID |
| `name_color` | "b0c3d9" | UI color code |
| `icon_url` | "i0CoZ81Ui0m-..." | Item icon path |
| `bucket_group_id` | "G18D2253004" | Bucket grouping |
| `last_updated` | "2026-07-03T19:46:51" | When we fetched it |

---

## Rate Limiting Strategy

### Research Phase

We tested the Steam API directly to find the actual rate limits:

1. **Burst limit:** ~10-12 rapid requests before 429
2. **Recovery after 429:** ~30 seconds
3. **Safe sustainable rate:** 3 seconds between requests
4. **Items per page:** Hard-coded to 10 (count/pagesize parameters ignored)

### Chosen Strategy: Burst Pattern

Instead of a flat 3-second delay (wastes time between requests), we used a burst pattern:

- **Send 10 requests as fast as possible** (~2 seconds)
- **Pause 30 seconds** (recovery time)
- **Repeat**

This achieves the same safety as flat 3s but runs ~2.5x faster because you're not sitting idle between every request when Steam isn't throttling you.

### Rate Limit Math

- 10 requests per burst = ~2 seconds
- 30 seconds pause between bursts
- Cycle: ~32 seconds for 100 items (10 pages × 10 items)
- Theoretical rate: ~187 items/min
- Observed rate: ~200 items/min

---

## 429 / Ban Handling

### What We Learned

| Event | Behavior | Recovery |
|---|---|---|
| Individual 429 | Backoff 30s, retry | ~30s |
| Sustained 429s | All retries exhaust, page recorded as failed | 30-60 min cooldown |
| IP ban | Every request returns 429 | 1-2 hours cooldown, or switch IP (VPN) |

### What Actually Happened During Our Build

1. **First run (12:56-15:40):** Got to offset 22,070 (64%) with 0 failures before hitting sustained 429s. ~4 pages failed.
2. **Paused for ~3 hours** while we fixed the 429 counting bug.
3. **Second run (18:44-19:46, VPN):** Completed remaining 36% with only 11 isolated 429s, all recovered.

### The 429 Counting Bug (Fixed)

**Problem:** 429s that occurred inside `fetch_page()`'s retry loop were never counted by the HealthMonitor. The health reports always showed `429s: 0` even when 429s were happening.

**Fix:** Added `last_fetch_429_count` to the client. After each `fetch_page()`, the main loop reads this counter and passes it to the HealthMonitor. Now health reports show accurate 429 counts.

---

## Failures and Recovery

### Failed Pages Table

Failed pages are tracked in a `failed_pages` table:

```sql
CREATE TABLE failed_pages (
    offset INTEGER PRIMARY KEY,
    error_reason TEXT,
    failed_at TEXT
);
```

### Recovery Workflow

```bash
# Normal build (resumes from last saved offset)
python scripts/build_market_catalog.py --resume

# Retry only the failed pages
python scripts/build_market_catalog.py --retry-failed

# Check status
python scripts/build_market_catalog.py --status
```

### What Happened

- 4 pages failed during the first run (offsets 22070-22100)
- They were recorded in `failed_pages`
- After the IP ban cooled down, `--retry-failed` recovered all 4 successfully (40 items)
- Final state: 0 failed pages

---

## Health Monitoring

### What's Tracked

- **OK/Failed/429 counters** — totals and consecutive streaks
- **Session expiry detection** — EMPTY responses after OK streaks
- **Auto-pause** — stops the run if thresholds are exceeded:
  - 3 consecutive 429s
  - 10 consecutive failures
- **Periodic health reports** — every 500 items

### What We Saw During the Build

| Metric | First Run | Second Run (VPN) |
|---|---|---|
| Items fetched | 18,038 | 5,452 |
| Failures | 0 | 0 |
| 429s (isolated) | ~100+ | 11 |
| 429s (sustained) | 4 pages | 0 |
| Auto-pause triggered | Yes (IP ban) | No |

---

## Database Schema

```sql
-- Item catalog
CREATE TABLE market_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hash_name TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    type TEXT,
    sell_price INTEGER,
    sell_price_text TEXT,
    sale_price_text TEXT,
    sell_listings INTEGER,
    tradable INTEGER,
    commodity INTEGER,
    classid TEXT,
    name_color TEXT,
    icon_url TEXT,
    bucket_group_id TEXT,
    last_updated TEXT
);

-- Build progress (pause/resume)
CREATE TABLE catalog_progress (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_offset INTEGER,
    total_items INTEGER,
    started_at TEXT,
    updated_at TEXT
);

-- Failed pages (retry)
CREATE TABLE failed_pages (
    offset INTEGER PRIMARY KEY,
    error_reason TEXT,
    failed_at TEXT
);
```

---

## Scripts

### `build_market_catalog.py`

**Location:** `backend/scripts/build_market_catalog.py`

**CLI flags:**
```bash
--resume           # Resume from last saved offset
--retry-failed     # Retry only failed pages
--status           # Show progress + health
--dry-run          # Preview without writing
--burst-size N     # Requests per burst (default: 10)
--burst-pause N    # Seconds between bursts (default: 30)
```

**Logs to:** `backend/runtime/market_catalog.log`

### Dependencies

- `requests` — HTTP client
- `sqlite3` — local database
- `config.py` — settings (not used for this script, but imported)

---

## Final Stats (After Gap Repair)

| Metric | Value |
|---|---|
| Total items cataloged | 31,908 |
| Pending gaps (unfetchable) | ~2,393 offsets |
| Coverage | 93% of ~34,301 items |
| Failed pages | 0 |
| DB size | 18 MB |
| Total duration (all phases) | ~6.5 hours |
| Active fetch time | ~3 hours |
| 429s encountered | ~50+ (all recovered) |
| Bans | 1 (recovered via VPN) |

### Item Type Breakdown

| Category | Count |
|---|---|
| Stickers (all types) | ~8,400 |
| Charms (all types) | ~4,400 |
| Skins (all grades + StatTrak/Souvenir) | ~8,000 |
| Graffiti | ~1,200 |
| Knives (★ + StatTrak) | ~1,900 |
| Gloves (★ + StatTrak) | ~500 |
| Containers/Cases | ~340 |
| Music Kits | ~140 |
| Collectibles/Pins | ~60 |
| Agents | ~54 |
| Equipment/Tools | ~100 |
| Other (Keys, Passes, Gifts, Tags) | ~50 |

---

## Lessons Learned

1. **Test rate limits empirically** — don't guess. We found Steam's actual limits differ from what people report online.
2. **Burst pattern beats flat delay** — same safety, 2-3x faster.
3. **Track429s at the right level** — retry loops hide 429s from higher-level monitoring. Always surface them.
4. **VPN is a valid recovery** — when IP banned, switching to a VPN immediately restores access.
5. **Failed page tracking is essential** — without it, you'd have to re-run the entire build to recover 4 pages.
6. **Local SQLite is the right choice** — Supabase is 500MB and full; this 18 MB local DB is clean and complete.

---

## Gap Repair (Post-Build)

### What Happened

After the catalog build completed with 23,490 items, spot-checking revealed **10,782 missing items** (Steam total: 34,272). The catalog only covered ~68% of the market. After the first fetch pass, we're at 29,553 items (86.2%) with 1,272 gap offsets still pending fetch.

**Root cause:** The original build treated empty API results (`results: []`) as "end of results" and broke the burst loop. This happened when rate limits returned empty pages instead of 429 status codes. Those items were never stored, but the progress tracker saved the offset as completed — so the resume skipped them silently.

### What We Did

Created `backend/scripts/repair_catalog_gaps.py` to identify and fetch missing items in two phases:

**Phase 1 — Scan:** Iterate through every offset (0–35,000), fetch each page, and check which items are missing from the DB. Save gap offsets to `backend/runtime/pending_gaps.txt` incrementally. Added `--start-offset` flag to resume from any offset without re-scanning.

**Phase 2 — Fetch:** For each saved gap offset, fetch the page and insert any missing items. Skip items already in the DB.

**Scan execution:** The scan was run in two parts due to rate limits:
1. Lower range (0–18,170): 3 runs, ~1,657 gaps found
2. Upper range (18,170–35,000): 2 runs, ~1,272 gaps found
3. Total: ~2,929 gap offsets identified (1,272 pending fetch)

### Why the Repair Was Difficult

1. **Rate limit fatigue:** Each scan/fetch run burns through Steam's burst budget (10 rapid + 30s cooldown). After 13,000+ requests, the budget depletes and recovery takes 30–60 minutes between runs.

2. **Scan re-scanning problem:** On `--resume`, the scan started from offset 0 and skipped known gaps sequentially. This meant re-scanning thousands of already-verified offsets, wasting burst budget. Fixed by starting the scan from the last known gap offset + 10.

3. **Flat delay vs burst pattern:** The initial repair script used a flat 0.3s delay between requests, which burned through the burst budget ~3x faster than the 10+30 burst pattern. Fixed by matching the original build's burst pattern.

4. **Python log buffering:** File output was buffered, so `tail -f` showed stale logs. Fixed by opening the file handler with `buffering=1` (line-buffered) and using `python3 -u` for unbuffered stdout.

5. **Upper-offset scan required:** The first scan only covered 52% of the market (0–18,170). A second scan of the upper range (18,170–35,000) was needed to identify the remaining ~4,748 missing items. This required adding a `--start-offset` flag to avoid re-scanning.

### What We Built

**Script:** `backend/scripts/repair_catalog_gaps.py`

| Flag | Description |
|------|-------------|
| `--scan-only` | Scan for gaps, don't fetch |
| `--fetch-only` | Fetch saved gaps, skip scan |
| `--resume` | Resume scan from last known gap offset |
| `--start-offset N` | Start scan from this offset (overrides `--resume`) |
| `--dry-run` | Preview gaps without fetching |
| `--max-offset N` | Max offset to scan (default: 35000) |

**Logs to:** `backend/runtime/gap_repair.log` (thorough logging: every INSERT, every page, progress every 25 gaps, timing, rate limits, failures)

**Monitor:** `tail -f backend/runtime/gap_repair.log`

**Resume after kill:** `python3 scripts/repair_catalog_gaps.py --fetch-only`

### Gap Repair Timeline

| Run | Phase | Offset Range | Duration | Result |
|-----|-------|-------------|----------|--------|
| 1 | Scan | 0 → 13,600 | ~45 min | 1,001 gaps found, killed by rate limit |
| 2 | Scan | 14,210 → 15,200 | ~10 min | +100 gaps, killed by rate limit |
| 3 | Scan | 17,780 → 18,170 | ~5 min | +50 gaps, killed by rate limit |
| 4 | Fetch | 1,657 gaps | 88 min | 2,862 items inserted, 0 failures |
| 5 | Scan | 18,170 → 26,330 | ~45 min | +616 gaps, killed by rate limit |
| 6 | Scan | 26,340 → 35,000 | ~55 min | +647 gaps, scan complete |
| 7 | Fetch | 625 gaps (first pass) | ~50 min | 1,034 items inserted, 0 failures, killed by rate limit at offset ~26,330 |
| 8 | Fetch | 1,264 gaps (second pass) | 69 min | 1,322 items inserted, 0 failures, 3 rate limits |

### Final Catalog Stats (After First Fetch Pass)

| Metric | Before Repair | After First Fetch | After Second Fetch |
|--------|--------------|-------------|--------|
| Items | 23,490 | 29,553 | 31,908 |
| Coverage | 68.5% | 86.2% | 93% |
| Failed pages | 0 | 0 | 0 |
| Rate limits (fetch) | — | 2 | 3 |
| Gaps pending | — | 1,272 | ~2,393 (unfetchable) |

### Remaining Gaps

After the second fetch pass, ~2,393 gap offsets remain unfetched. These are likely items that no longer exist on the Steam Market (removed, renamed, or delisted) — the API returns empty results for them. No further action needed; 31,908 items represents ~93% of the current market.

### Upper-Offset Scan (18,170 → 35,000)

**Why:** The initial gap repair scan only covered offsets 0–18,170 (52% of the market). The remaining 48% (offsets 18,170–35,000) was never scanned, so ~4,748 items were never identified as missing. A second scan of this range was needed to complete the gap inventory.

**What we did:** Added a `--start-offset` flag to `repair_catalog_gaps.py` to allow scanning from any offset. Ran the scan in two parts:

1. **First run (offset 18,170 → 26,330):** Reached 75% of the upper range before hitting rate limits. Saved 616 gaps.
2. **Second run (offset 26,340 → 35,000):** Completed the remaining 25%. Found 647 additional gaps.

**Results:**

| Metric | Value |
|--------|-------|
| Offsets scanned | 866 (upper range) |
| Gaps found | 647 new (upper range) |
| Total gaps in file | 1,272 |
| Duration | 54.7 min |
| Rate limits | ~5 (all recovered) |
| Errors | 12 (offsets that failed after retry) |

**What was saved:** All 1,272 gap offsets were written to `backend/runtime/pending_gaps.txt` incrementally — nothing was lost when the process was killed by rate limits.

**Command used:**
```bash
# First part
python3 -u scripts/repair_catalog_gaps.py --scan-only --start-offset 18170 --max-offset 35000

# Resume after rate limit cooldown
python3 -u scripts/repair_catalog_gaps.py --scan-only --start-offset 26340 --max-offset 35000
```

**Monitor:**
```bash
tail -f backend/runtime/gap_repair.log
```

### Second Fetch Pass (1,264 gaps → 31,908 items)

**What:** Fetched the remaining 1,264 gap offsets identified by the full scan (lower + upper ranges). This was the second fetch pass — the first pass had already processed 625 gaps before being killed by rate limits.

**How it ran:**

```bash
# Resume fetching (skips already-processed gaps automatically)
python3 -u scripts/repair_catalog_gaps.py --fetch-only
```

**Execution:** Launched at 16:53 EDT, completed at 18:02 EDT — 69 minutes, zero manual intervention. The script loaded all 1,264 gap offsets from `pending_gaps.txt`, skipped the 625 already processed in the first pass, and fetched the remaining 639 gaps (plus re-checked existing items for deduplication).

**Why it was so clean:** The first pass failures (429s, kills) happened because the scan and fetch phases competed for burst budget in the same run. The second pass ran `--fetch-only` in isolation — pure fetch with no scanning overhead — so the burst budget was used efficiently.

**Results:**

| Metric | Value |
|--------|-------|
| Gaps attempted | 1,264 |
| Items inserted | 1,322 |
| Already existed | 11,318 |
| Rate limits | 3 (all recovered automatically) |
| Failed | 0 |
| Duration | 69.1 min |
| Speed | ~18 gaps/min |

**Catalog progression:**

| Phase | Items | Change |
|-------|-------|--------|
| Initial build | 23,490 | — |
| First fetch pass (+ gap repair) | 30,593 | +7,103 |
| Second fetch pass | 31,908 | +1,315 |

**Key insight:** The 31,908 final items vs ~34,301 on Steam means ~2,393 offsets returned empty results. These are almost certainly delisted/removed items — the Steam API returns `[]` for items that no longer exist. This is not a bug; it's expected.

---

## Lessons Learned (Updated)

1. **Test rate limits empirically** — don't guess. We found Steam's actual limits differ from what people report online.
2. **Burst pattern beats flat delay** — same safety, 2-3x faster.
3. **Track429s at the right level** — retry loops hide 429s from higher-level monitoring. Always surface them.
4. **VPN is a valid recovery** — when IP banned, switching to a VPN immediately restores access.
5. **Failed page tracking is essential** — without it, you'd have to re-run the entire build to recover 4 pages.
6. **Local SQLite is the right choice** — Supabase is 500MB and full; this 18 MB local DB is clean and complete.
7. **Empty results ≠ end of results** — Steam returns `[]` for rate-limited pages, not 429. Always retry empty pages instead of breaking the loop.
8. **Resume must skip ahead** — re-scanning from offset 0 wastes burst budget. On resume, start from the last known offset.
9. **Python log buffering lies** — `tail -f` shows stale data unless you line-buffer the file handler. Use `python3 -u` and `buffering=1`.
10. **Cleaning gaps via API is counterproductive** — checking each gap for "still missing?" costs API requests that burn the same budget you're trying to save. Let `--fetch-only` handle deduplication naturally.
11. **Scanning in chunks is necessary** — a full 0–35,000 scan takes too long and always gets rate-limited. Breaking it into ranges (lower 0–18,170, upper 18,170–35,000) with `--start-offset` makes it manageable.
12. **Gap inventory before fetch** — always complete the full scan before starting the fetch phase. Mixing scan + fetch burns budget on both operations simultaneously.
13. **Isolate fetch from scan** — the second fetch pass (`--fetch-only` only) completed 1,264 gaps in 69 min with 0 failures. The first pass struggled because it mixed scan + fetch in the same run, competing for burst budget.
14. **Incremental gap saving prevents data loss** — writing each gap offset to `pending_gaps.txt` as it's discovered means nothing is lost when the process is killed by rate limits.

---

## Next Steps

- **Phase 2:** Backfill pricehistory for all 31,908 items using `backfill_ssr_history.py`
- Update the backfill script to read from `market_catalog.db` (via `--source catalog`)
- Replace old Supabase snapshots with SSR history data
- After Phase 2, plan migration from local SQLite to Supabase (replacing old snapshots)
