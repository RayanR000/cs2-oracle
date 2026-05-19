# Phase 2 Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get real Steam API prices flowing hourly into the database with 100+ items and 365 days of synthetic history context.

**Architecture:** 
1. Add hash name resolution to steam_market.py so it can properly query Steam API
2. Integrate the comprehensive data source files (cs2_data_sources.py, comprehensive_loader.py) 
3. Update main.py to seed 100+ items + 1 year history on startup
4. Update pipeline.py to use improved collector
5. End-to-end test: real data flowing

**Tech Stack:** FastAPI, SQLAlchemy, Supabase PostgreSQL, APScheduler, requests (Steam API)

---

## File Structure

**Creating/Adding:**
- `backend/collectors/cs2_data_sources.py` (track in git - already written)
- `backend/collectors/comprehensive_loader.py` (track in git - already written)

**Modifying:**
- `backend/collectors/steam_market.py` - Add hash name resolution
- `backend/main.py` - Call comprehensive_loader on startup
- `backend/collectors/__init__.py` - Export new modules

**Verifying:**
- `backend/collectors/pipeline.py` - Ensure it uses hash names properly
- `backend/database.py` - Verify Item/PriceHistory models

---

## Tasks

### Task 1: Add untracked files to git and update collectors __init__.py

**Files:**
- Add: `backend/collectors/cs2_data_sources.py` (already written, untracked)
- Add: `backend/collectors/comprehensive_loader.py` (already written, untracked)
- Modify: `backend/collectors/__init__.py`

- [ ] **Step 1: Check collectors __init__.py exists**

```bash
ls -la backend/collectors/__init__.py
```

If it doesn't exist, create it as empty file.

- [ ] **Step 2: Add exports to collectors __init__.py**

Read the file first, then update it:

```python
from .steam_market import SteamMarketCollector, MockSteamMarketCollector
from .data_validation import DataValidator, DataCleaner
from .pipeline import DataPipeline, PipelineMonitor
from .cs2_data_sources import CS2ItemCatalog, CS2GameEvents, HistoricalDataGenerator
from .comprehensive_loader import ComprehensiveDataLoader, load_all_cs2_data

__all__ = [
    'SteamMarketCollector',
    'MockSteamMarketCollector',
    'DataValidator',
    'DataCleaner',
    'DataPipeline',
    'PipelineMonitor',
    'CS2ItemCatalog',
    'CS2GameEvents',
    'HistoricalDataGenerator',
    'ComprehensiveDataLoader',
    'load_all_cs2_data',
]
```

- [ ] **Step 3: Stage and commit both new files**

```bash
cd backend
git add collectors/cs2_data_sources.py collectors/comprehensive_loader.py collectors/__init__.py
git commit -m "feat: add CS2 data sources and comprehensive data loader

- cs2_data_sources.py: complete item catalog (100+ items), game events, synthetic history generator
- comprehensive_loader.py: bulk loader for catalog and historical data seeding
- Export from collectors package for easy imports

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

Expected: Commit succeeds, both files now tracked.

---

### Task 2: Enhance steam_market.py with hash name resolution

**Files:**
- Modify: `backend/collectors/steam_market.py` (around L14-221)

**Problem:** Current code doesn't handle Steam's market hash names. Steam API requires:
- Hash format: `"AK-47%20%7C%20Phantom%20Disruptor-Factory%20New"` 
- Not: `"AK-47 | Phantom Disruptor"`

**Solution:** Query Steam's market listing endpoint to resolve item names to hash names, cache the results.

- [ ] **Step 1: Read steam_market.py to understand current structure**

```bash
head -50 backend/collectors/steam_market.py
```

Review the SteamMarketCollector.__init__ and existing methods.

- [ ] **Step 2: Add hash name cache to __init__**

In the `__init__` method (after `self.rate_limit_delay = rate_limit_delay`), add:

```python
self.hash_name_cache = {}  # Maps item_name -> market_hash_name
```

- [ ] **Step 3: Add hash name resolver method**

Add this new method to the SteamMarketCollector class (after `_make_request`, before `get_item_price_history`):

```python
def resolve_hash_name(self, item_name: str) -> Optional[str]:
    """
    Resolve item name to Steam market hash name.
    Uses Steam's market search endpoint to find the exact hash.
    Results are cached to avoid repeated lookups.
    """
    # Check cache first
    if item_name in self.hash_name_cache:
        return self.hash_name_cache[item_name]
    
    # Query Steam market search endpoint
    url = "https://steamcommunity.com/market/search/render/"
    params = {
        'query': item_name,
        'start': 0,
        'count': 10,
        'search_descriptions': 0,
        'sort_column': 'name',
        'sort_dir': 'asc'
    }
    
    data = self._make_request(url, params)
    if not data or 'results' not in data or not data['results']:
        logging.warning(f"No market hash found for: {item_name}")
        return None
    
    # Extract hash name from first result
    first_result = data['results'][0]
    hash_name = first_result.get('hash_name')
    
    if hash_name:
        # Cache it
        self.hash_name_cache[item_name] = hash_name
        logging.debug(f"Resolved {item_name} -> {hash_name}")
        return hash_name
    
    logging.warning(f"Could not extract hash_name from result for: {item_name}")
    return None
```

- [ ] **Step 4: Update get_item_price_history to use hash names**

Find the `get_item_price_history` method. Replace the entire method with:

```python
def get_item_price_history(self, item_name_or_hash: str) -> Optional[Tuple[float, int, datetime]]:
    """
    Get current price and volume for an item.
    Accepts either item name or market hash name.
    """
    # If it doesn't look like a hash name, resolve it first
    if '%' not in item_name_or_hash:
        hash_name = self.resolve_hash_name(item_name_or_hash)
        if not hash_name:
            return None
    else:
        hash_name = item_name_or_hash
    
    # Query price history endpoint
    url = "https://steamcommunity.com/market/pricehistory/"
    params = {
        'country': 'US',
        'currency': 1,  # USD
        'appid': 730,  # CS2
        'market_hash_name': hash_name
    }
    
    data = self._make_request(url, params)
    if not data or 'prices' not in data or not data['prices']:
        logging.warning(f"No price data for: {hash_name}")
        return None
    
    # Get most recent price point (last in list)
    prices = data['prices']
    last_price_point = prices[-1]  # [timestamp_str, price_str, volume_str]
    
    try:
        price = float(last_price_point[1])
        volume = int(last_price_point[2])
        return (price, volume, datetime.now())
    except (ValueError, IndexError) as e:
        logging.error(f"Error parsing price data: {e}")
        return None
```

- [ ] **Step 5: Run the Steam market tests to check for immediate issues**

```bash
cd backend
python -m pytest tests/ -v -k steam 2>&1 | head -50
```

Expected: Tests may fail or be skipped (if none exist), that's OK. We're checking for import/syntax errors.

- [ ] **Step 6: Commit the enhancements**

```bash
cd backend
git add collectors/steam_market.py
git commit -m "feat: add hash name resolution to Steam market collector

- Add resolve_hash_name() method to convert item names to market hash names
- Cache hash names to avoid repeated API lookups
- Update get_item_price_history to accept both names and hashes
- Use Steam market search API to find exact hash for each item

This fixes data collection failures caused by name format mismatches.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

---

### Task 3: Update main.py to load catalog and history on startup

**Files:**
- Modify: `backend/main.py` (around startup event handlers)

**Current state:** main.py initializes DB and seeds 8 sample items. We need to replace that with comprehensive_loader.

- [ ] **Step 1: Read main.py to see current startup flow**

```bash
cat backend/main.py | head -80
```

Note the `@app.on_event("startup")` section.

- [ ] **Step 2: Update imports in main.py**

Add this import near the top after other collector imports:

```python
from collectors.comprehensive_loader import load_all_cs2_data
```

- [ ] **Step 3: Replace the startup event handler**

Find the `@app.on_event("startup")` section. Replace the entire function with:

```python
@app.on_event("startup")
async def startup():
    """Initialize database and load data on startup"""
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized successfully")
    
    logger.info("Loading CS2 catalog and synthetic history...")
    try:
        stats = load_all_cs2_data()
        logger.info(f"Data load complete: {stats}")
        logger.info(f"  Items: {stats.get('items_added', 0)} added, {stats.get('items_skipped', 0)} skipped")
        logger.info(f"  Price records: {stats.get('price_records_added', 0)}")
        logger.info(f"  Events: {stats.get('events_added', 0)}")
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
    
    logger.info("Starting real-time market data collection...")
    global pipeline
    pipeline = DataPipeline()
    pipeline.start()
    logger.info("Real-time data collection started")
```

- [ ] **Step 4: Remove old seed_data import if present**

Look for any `from seed_data import` statements and remove them (they're no longer needed).

- [ ] **Step 5: Test the imports work**

```bash
cd backend
python -c "from collectors.comprehensive_loader import load_all_cs2_data; print('Import OK')"
```

Expected: `Import OK`

- [ ] **Step 6: Commit the changes**

```bash
cd backend
git add main.py
git commit -m "feat: load complete catalog on startup via comprehensive_loader

- Call load_all_cs2_data() on app startup
- Loads 100+ items from CS2ItemCatalog
- Generates 365 days of synthetic history for each item
- Loads game events
- Logs detailed statistics
- Replaces old 8-item seed approach

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

---

### Task 4: Verify pipeline.py uses improved steam_market collector

**Files:**
- Check: `backend/collectors/pipeline.py` (around L84-171, run_daily_collection method)

**Goal:** Ensure the pipeline's data collection uses the new hash name resolver.

- [ ] **Step 1: Read run_daily_collection method**

```bash
sed -n '84,171p' backend/collectors/pipeline.py
```

Find where it calls `self.collector.get_item_price_history()` or similar.

- [ ] **Step 2: Verify it passes item names, not hash names**

The method should iterate over items and call something like:

```python
for item in items:
    result = self.collector.get_item_price_history(item.name)
```

If the code passes `item.name` (not a hash), it's correct. The improved steam_market.py will handle the hash resolution automatically.

If it passes something else, note it but don't change (the new collector is backwards-compatible).

- [ ] **Step 3: Check error handling**

Look for try/except blocks around collector calls. They should exist. If they don't, add one:

```python
try:
    result = self.collector.get_item_price_history(item.name)
    if result:
        # Process result
    else:
        logger.warning(f"No data for {item.name}")
except Exception as e:
    logger.error(f"Error collecting {item.name}: {e}")
    continue
```

If this already exists, you can skip this step.

- [ ] **Step 4: Commit if changes made**

If you made changes, run:

```bash
cd backend
git add collectors/pipeline.py
git commit -m "fix: add error handling to pipeline collection loop

Ensure collection failures don't crash the pipeline, log warnings instead.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

If no changes needed, no commit required.

---

### Task 5: Test end-to-end data flow

**Files:**
- Test: Manual verification via API and logs

**Goal:** Start the backend and verify:
1. 100+ items load
2. Synthetic history created
3. Real data collection starts
4. API returns data

- [ ] **Step 1: Start the backend**

```bash
cd backend
source venv/bin/activate
python main.py 2>&1 | tee startup.log &
sleep 5
```

This starts the server in the background and saves logs to startup.log.

- [ ] **Step 2: Verify 100+ items loaded in the logs**

```bash
grep -i "items" startup.log | grep -i "added\|loaded"
```

Expected output should show something like:
```
Data load complete: {'items_added': 100+, ...}
```

If you see "items_skipped: 100" instead, it means items already exist in the DB (which is fine).

- [ ] **Step 3: Check real data collection started**

```bash
grep -i "collection" startup.log | head -10
```

Expected: Lines mentioning "Started server process", "Starting real-time market data collection", "Background data collection started"

- [ ] **Step 4: Query the items API to verify data loads**

```bash
curl -s http://localhost:8000/items/ | python -m json.tool | head -50
```

Expected: JSON array with 100+ items. Check that you see items like:
- "AK-47 | Neon Rider"
- "Karambit | Doppler"
- "CS2 Weapon Case"

- [ ] **Step 5: Get a specific item's price history**

```bash
# Get first item ID from the items list
ITEM_ID=$(curl -s http://localhost:8000/items/?limit=1 | python -c "import sys, json; data=json.load(sys.stdin); print(data['items'][0]['id'])" 2>/dev/null)

# Query price history
curl -s http://localhost:8000/items/$ITEM_ID/price-history | python -m json.tool | head -30
```

Expected: JSON with array of price records. You should see dates spanning ~365 days.

- [ ] **Step 6: Trigger manual collection and wait for new data**

```bash
curl -X POST http://localhost:8000/admin/collect-now

# Wait 5 seconds
sleep 5

# Check collection status
curl -s http://localhost:8000/admin/collection-status | python -m json.tool
```

Expected: collection_status should show recent timestamp, with some "successful" items (if Steam API is working) or all "failed" (if there are API issues, which is OK for now—we're verifying structure).

- [ ] **Step 7: Check the server logs for collection results**

```bash
tail -50 startup.log | grep -i "collection\|successful\|failed"
```

Expected: Lines like "Collection complete: X successful, Y failed" or "Running scheduled data collection"

- [ ] **Step 8: Stop the server**

```bash
pkill -f "python main.py"
```

---

### Task 6: Commit final state and document results

**Files:**
- No code changes, just verification and documentation

- [ ] **Step 1: Check git status**

```bash
cd backend
git status
```

Expected: All untracked files should now be staged and committed. Working directory should be clean (nothing to commit).

- [ ] **Step 2: View the commit log to confirm all changes are there**

```bash
git log --oneline -5
```

Expected: Last 5 commits should be:
1. Commit results doc (from Task 6)
2. Fix pipeline error handling (from Task 4, if needed)
3. Load catalog on startup (from Task 3)
4. Hash name resolution in steam_market (from Task 2)
5. Add CS2 data sources and loader (from Task 1)

- [ ] **Step 3: Create a Phase 2 completion summary**

Create `docs/PHASE2_COMPLETION.md`:

```markdown
# Phase 2 Completion Summary

**Completed**: 2026-05-19

## What Was Implemented

1. **Steam Market Collector Enhancement** - Added hash name resolution
   - Resolves item names to Steam's market hash format
   - Caches results to minimize API calls
   - Properly queries Steam's price history endpoint

2. **Data Source Integration** - Incorporated comprehensive data sources
   - cs2_data_sources.py: 100+ item catalog, game events, synthetic history generator
   - comprehensive_loader.py: Bulk loading with idempotent operations

3. **Startup Data Seeding** - App now loads full catalog on startup
   - 100+ items automatically loaded into Item table
   - 365 days of synthetic price history generated per item
   - Game events loaded
   - Takes ~30-60 seconds on first run, skips on subsequent runs

4. **Hourly Data Collection** - Real Steam data collection active
   - Scheduled every 3600 seconds (1 hour)
   - Uses improved steam_market collector with hash name resolution
   - Logs collection stats (items collected, failures, timing)
   - Continues even if some items fail

## How to Verify

```bash
# Start the backend
cd backend && python main.py

# Check logs for 100+ items loaded and synthetic history
# Should see: "Data load complete: {'items_added': ..., 'price_records_added': ...}"

# Query API
curl http://localhost:8000/items/ | jq '.items | length'  # Should be 100+

# Trigger manual collection
curl -X POST http://localhost:8000/admin/collect-now

# Check collection results
curl http://localhost:8000/admin/collection-status | jq .
```

## Next Steps (Phase 3)

- Build frontend UI with charts and dashboards
- Improve trend analysis algorithms
- Add prediction capabilities
- Implement opportunity detection

## Known Limitations

- Steam API data collection may fail for some items due to Steam API restrictions
- Synthetic historical data is realistic but not real market data (only current data is real)
- Error handling logs warnings but continues—no alerting system yet
```

Add and commit:

```bash
git add docs/PHASE2_COMPLETION.md
git commit -m "docs: Phase 2 completion summary and verification guide

Phase 2 is complete:
- Real data collection pipeline running
- 100+ items loaded on startup
- Hash name resolution fixes Steam API integration
- Synthetic history provides market context

Includes verification steps and next steps for Phase 3.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Final verification**

```bash
git log --oneline | head -10
```

All Phase 2 commits should be present.

---

## Self-Review

**Spec coverage:**
- ✅ Fix steam_market.py hash names → Task 2
- ✅ Integrate cs2_data_sources.py → Task 1
- ✅ Integrate comprehensive_loader.py → Task 1
- ✅ Update main.py startup → Task 3
- ✅ Update pipeline.py → Task 4
- ✅ Verify real data flows → Task 5
- ✅ Success criteria documented → Task 5 and 6

**Placeholder scan:**
- No "TBD", "TODO", or "fill in" anywhere
- All code blocks complete with actual implementation
- All commands include expected output

**Type consistency:**
- `item_name` consistently used for item names
- `hash_name` for Steam market hash format
- Return types consistent (Optional[Tuple], Optional[Dict], etc.)

**Gaps identified:** None—spec fully covered.
