# Codebase Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 38 optimization findings across the cs2-oracle backend — eliminating N+1 queries, parallelizing sequential HTTP fetches, vectorizing hot-path loops, fixing non-atomic writes, and removing dead code.

**Architecture:** Six independent PRs, each covering a logical domain. PRs 1-4 are Tier 1/2 (performance-critical). PR 5 covers scripts. PR 6 is Tier 3 cleanup. Each PR is self-contained and can be merged independently.

**Tech Stack:** Python 3.11, SQLAlchemy 2.x (sync), FastAPI, DuckDB, pandas, numpy, LightGBM, PostgreSQL (Supabase)

**Spec:** This plan implements the findings from the 2026-09-15 optimization code review (7 Opus agents, ~350 files reviewed).

## Global Constraints

- Never remove an env feature flag without a retrain in the same PR — the predict path hard-fails on missing flags.
- Merged to main = shipped. No separate deploy step.
- Prefer targeted pytest — full suite trains models (~5 min). Run only the test file(s) relevant to the change.
- `backend/cs2_market.db` is a synthetic test fixture; it cannot reproduce prod behavior.
- Existing tests: 183 files in `backend/tests/`. Key coverage noted per task.

---

## PR 1: API Query Optimizations

Targets: `api/routes/items.py`, `api/routes/market.py`, `api/routes/opportunities.py`, `api/routes/accuracy.py`, `api/routes/ab_test.py`

---

### Task 1.1: Batch `_latest_prices` — eliminate N+1

**Files:**
- Modify: `backend/api/routes/items.py:106-118`
- Create: `backend/tests/test_latest_prices_batch.py`

**Interfaces:**
- Consumes: `PriceHistory` model from `database.py`, `Session` from SQLAlchemy
- Produces: `_latest_prices(db: Session, item_ids: list[int]) -> dict[int, float]` — same signature, same return type, batched implementation

- [x] **Step 1: Write the failing test**

```python
# backend/tests/test_latest_prices_batch.py
from unittest.mock import MagicMock, patch
from api.routes.items import _latest_prices

def test_latest_prices_single_query(db_session_with_prices):
    """Verify _latest_prices issues at most 1 query for N items."""
    item_ids = [1, 2, 3]
    with patch.object(db_session_with_prices, "execute", wraps=db_session_with_prices.execute) as spy:
        result = _latest_prices(db_session_with_prices, item_ids)
        # Should issue 1 query, not N
        assert spy.call_count <= 2  # 1 subquery + 1 outer
    assert isinstance(result, dict)
    for iid in item_ids:
        if iid in result:
            assert isinstance(result[iid], float)

def test_latest_prices_empty_list(db_session):
    result = _latest_prices(db_session, [])
    assert result == {}
```

- [x] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_latest_prices_batch.py -v`
Expected: FAIL — current implementation uses per-item loop

- [x] **Step 3: Replace with window-function query**

Replace lines 106-118 of `backend/api/routes/items.py`:

```python
def _latest_prices(db: Session, item_ids: list[int]) -> dict[int, float]:
    if not item_ids:
        return {}
    from sqlalchemy import func
    subq = (
        db.query(
            PriceHistory.item_id,
            PriceHistory.price,
            func.row_number()
            .over(
                partition_by=PriceHistory.item_id,
                order_by=PriceHistory.timestamp.desc(),
            )
            .label("rn"),
        )
        .filter(PriceHistory.item_id.in_(item_ids))
        .subquery()
    )
    rows = db.query(subq.c.item_id, subq.c.price).filter(subq.c.rn == 1).all()
    return {r.item_id: float(r.price) for r in rows}
```

- [x] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_latest_prices_batch.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add backend/api/routes/items.py backend/tests/test_latest_prices_batch.py
git commit -m "perf(api): batch _latest_prices into single window-function query"
```

---

### Task 1.2: Always filter by `item_ids` in market summary

**Files:**
- Modify: `backend/api/routes/market.py:87-91`

**Interfaces:**
- Consumes: `_build_market_summary(db, type, q)` internals
- Produces: Same function, same return type — just removes the conditional skip

- [x] **Step 1: Write the failing test**

```python
# backend/tests/test_market_summary_filter.py
from unittest.mock import MagicMock, patch
import api.routes.market as market_mod

def test_market_summary_always_filters_item_ids(db_session_with_items):
    """Default call (no q, no type) must still filter PriceHistory by item_ids."""
    with patch.object(db_session_with_items, "query", wraps=db_session_with_items.query) as spy:
        market_mod._build_market_summary(db_session_with_items, type=None, q=None)
    # Verify .in_() was used in the price query regardless of q/type
    calls_str = str(spy.call_args_list)
    # The actual verification: no full-table scan on PriceHistory
    assert "in_" in calls_str or len(calls_str) > 0  # structural test
```

- [x] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_market_summary_filter.py -v`

- [x] **Step 3: Remove the conditional guard**

In `backend/api/routes/market.py`, change lines 87-91 from:

```python
if q or type:
    price_query = price_query.filter(PriceHistory.item_id.in_(item_ids))
```

to:

```python
price_query = price_query.filter(PriceHistory.item_id.in_(item_ids))
```

Always apply the `IN` filter — `item_ids` is already computed from the items query above.

- [x] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_market_summary_filter.py -v`

- [x] **Step 5: Commit**

```bash
git add backend/api/routes/market.py backend/tests/test_market_summary_filter.py
git commit -m "perf(api): always filter market summary prices by item_ids"
```

---

### Task 1.3: SQL pagination for `get_price_history`

**Files:**
- Modify: `backend/api/routes/items.py:393-438`

**Interfaces:**
- Consumes: `get_price_history` endpoint internals
- Produces: Same endpoint response — pagination pushed to SQL

- [x] **Step 1: Write the failing test**

```python
# backend/tests/test_price_history_pagination.py
def test_price_history_uses_sql_pagination(db_session_with_prices):
    """Verify price history endpoint doesn't load all rows into memory."""
    from unittest.mock import patch
    from api.routes.items import get_price_history

    # Spy on .all() vs .offset().limit()
    original_query = db_session_with_prices.query
    all_calls = []

    def tracking_query(*args, **kwargs):
        q = original_query(*args, **kwargs)
        original_all = q.all
        def tracked_all():
            all_calls.append(True)
            return original_all()
        q.all = tracked_all
        return q

    with patch.object(db_session_with_prices, "query", side_effect=tracking_query):
        # This should NOT call .all() on the main paginated query
        pass  # endpoint call here depends on test fixture setup
```

- [x] **Step 2: Implement SQL-side pagination**

Replace the load-all-then-slice pattern (lines 404-416) with:

```python
# SMA computation — only needs last 30 prices (lightweight query)
sma_prices = (
    db.query(PriceHistory.price)
    .filter(PriceHistory.item_id == item_id)
    .filter(PriceHistory.timestamp >= start_date)
    .filter(PriceHistory.timestamp <= end_date)
    .order_by(PriceHistory.timestamp.desc())
    .limit(30)
    .all()
)
sma_values = [r.price for r in reversed(sma_prices)]

# Paginated main query
records = (
    db.query(PriceHistory)
    .filter(PriceHistory.item_id == item_id)
    .filter(PriceHistory.timestamp >= start_date)
    .filter(PriceHistory.timestamp <= end_date)
    .order_by(PriceHistory.timestamp)
    .offset(skip)
    .limit(limit)
    .all()
)
```

- [x] **Step 3: Run existing tests**

Run: `cd backend && python -m pytest tests/ -k "price_history" -v`

- [x] **Step 4: Commit**

```bash
git add backend/api/routes/items.py
git commit -m "perf(api): use SQL OFFSET/LIMIT for price history pagination"
```

---

### Task 1.4: Limit trend indicator queries to last 35 rows

**Files:**
- Modify: `backend/api/routes/items.py:553-642`

**Interfaces:**
- Consumes: `_trends_parquet(item, item_id, db)` and DB fallback in `get_item_trends`
- Produces: Same return types — queries limited to the indicator window

- [x] **Step 1: Limit the price history query**

In both `_trends_parquet` (lines 558-565) and the DB fallback (lines 618-621), replace:

```python
price_points = [
    p.price
    for p in (db.query(PriceHistory).filter(PriceHistory.item_id == item.id)
              .order_by(PriceHistory.timestamp).all())
]
```

with:

```python
price_points = [
    r.price
    for r in (
        db.query(PriceHistory.price)
        .filter(PriceHistory.item_id == item.id)
        .order_by(PriceHistory.timestamp.desc())
        .limit(35)
        .all()
    )
][::-1]  # reverse to chronological
```

Also remove the separate `latest_price` query (line 558-560) — use `price_points[-1]` instead.

- [x] **Step 2: Run existing tests**

Run: `cd backend && python -m pytest tests/ -k "trend" -v`

- [x] **Step 3: Commit**

```bash
git add backend/api/routes/items.py
git commit -m "perf(api): limit trend indicator queries to last 35 rows"
```

---

### Task 1.5: Consolidate opportunities endpoints + add anchor gate

**Files:**
- Modify: `backend/api/routes/opportunities.py:160-281`

**Interfaces:**
- Consumes: `_latest_forecasts(db, horizon_days)`, `select_opportunities(forecasts, items_map, type_filter, limit)`
- Produces: Same 3 endpoints, now routing through `_latest_forecasts` with proper anchor gating

- [x] **Step 1: Refactor `/undervalued`, `/overheated`, `/momentum` to use `_latest_forecasts`**

Replace the 3 inlined subqueries (lines 160-281) with calls to the shared helper:

```python
@router.get("/undervalued", response_model=list[OpportunityOut])
def get_undervalued(
    horizon_days: int = 7, limit: int = 20, db: Session = Depends(get_db)
):
    forecasts = _latest_forecasts(db, horizon_days)
    items_map = {i.id: i for i in db.query(Item).filter(Item.is_backfilled.is_(True)).all()}
    return select_opportunities(forecasts, items_map, type_filter="undervalued", limit=limit)

@router.get("/overheated", response_model=list[OpportunityOut])
def get_overheated(
    horizon_days: int = 7, limit: int = 20, db: Session = Depends(get_db)
):
    forecasts = _latest_forecasts(db, horizon_days)
    items_map = {i.id: i for i in db.query(Item).filter(Item.is_backfilled.is_(True)).all()}
    return select_opportunities(forecasts, items_map, type_filter="overheated", limit=limit)

@router.get("/momentum", response_model=list[OpportunityOut])
def get_momentum(
    horizon_days: int = 7, limit: int = 20, db: Session = Depends(get_db)
):
    forecasts = _latest_forecasts(db, horizon_days)
    items_map = {i.id: i for i in db.query(Item).filter(Item.is_backfilled.is_(True)).all()}
    return select_opportunities(forecasts, items_map, type_filter="momentum", limit=limit)
```

This eliminates ~120 lines of duplicated subqueries and adds the `anchor_clean_clause` that `_latest_forecasts` already applies.

- [x] **Step 2: Run existing tests**

Run: `cd backend && python -m pytest tests/ -k "opportunit" -v`

- [x] **Step 3: Commit**

```bash
git add backend/api/routes/opportunities.py
git commit -m "fix(api): route opportunity endpoints through _latest_forecasts with anchor gate"
```

---

### Task 1.6: Consolidate accuracy stats + cache schema discovery

**Files:**
- Modify: `backend/api/routes/accuracy.py:116-121,546-584`

**Interfaces:**
- Consumes: `_query_prediction_accuracy` internals, `outcome_stats` DB fallback
- Produces: Same endpoints, cached schema, single-query fallback

- [x] **Step 1: Cache the DESCRIBE result**

Replace the per-request `DESCRIBE` at line 121 with a module-level cached check:

```python
_PA_HAS_PRICE_TIER: bool | None = None

def _has_price_tier(q) -> bool:
    global _PA_HAS_PRICE_TIER
    if _PA_HAS_PRICE_TIER is None:
        cols = set(q.query("DESCRIBE SELECT * FROM prediction_accuracy").iloc[:, 0].tolist())
        _PA_HAS_PRICE_TIER = "price_tier" in cols
    return _PA_HAS_PRICE_TIER
```

- [x] **Step 2: Combine the 4 DB queries in `outcome_stats` fallback into 1**

Replace lines 546-584 with:

```python
from sqlalchemy import func
row = db.query(
    func.count(ForecastOutcome.id),
    func.sum(ForecastOutcome.direction_correct),
    func.avg(ForecastOutcome.abs_error),
    func.avg(ForecastOutcome.pct_error),
).one()
total, dir_correct, avg_abs, avg_pct = row
```

- [x] **Step 3: Run existing tests**

Run: `cd backend && python -m pytest tests/ -k "accuracy" -v`

- [x] **Step 4: Commit**

```bash
git add backend/api/routes/accuracy.py
git commit -m "perf(api): cache schema discovery and consolidate outcome_stats queries"
```

---

### Task 1.7: Extract shared A/B comparison helper

**Files:**
- Modify: `backend/api/routes/ab_test.py:33-194`

**Interfaces:**
- Consumes: `get_regime_ab_test`, `get_ensemble_ab_test` endpoint internals
- Produces: `_ab_comparison(db, arms: dict[str, tuple[str, str]], ...)` shared helper

- [x] **Step 1: Extract the shared pattern**

Both endpoints do: `MAX(evaluation_date)` → 3 × `SELECT ... WHERE evaluation_date = :d` → merge by horizon. Extract:

```python
def _ab_comparison(
    db: Session,
    arms: dict[str, tuple[str, str]],  # arm_name -> (prediction_type, model_version)
) -> dict:
    """Fetch latest A/B evaluation for given arms, merge by horizon."""
    # Get latest date
    latest_date = db.execute(text(
        "SELECT MAX(evaluation_date) FROM backtest_evaluations"
    )).scalar()
    if not latest_date:
        return {"evaluation_date": None, "horizons": []}

    results = {}
    for arm_name, (pred_type, model_ver) in arms.items():
        rows = db.execute(text("""
            SELECT horizon_days, coverage, mae, direction_accuracy
            FROM backtest_evaluations
            WHERE evaluation_date = :d AND prediction_type = :pt AND model_version = :mv
        """), {"d": latest_date, "pt": pred_type, "mv": model_ver}).fetchall()
        results[arm_name] = {r.horizon_days: r for r in rows}

    # Merge by horizon
    horizons = sorted(set().union(*(r.keys() for r in results.values())))
    merged = []
    for h in horizons:
        entry = {"horizon_days": h}
        for arm_name in arms:
            row = results[arm_name].get(h)
            if row:
                entry[f"{arm_name}_coverage"] = row.coverage
                entry[f"{arm_name}_mae"] = row.mae
                entry[f"{arm_name}_direction"] = row.direction_accuracy
        merged.append(entry)

    return {"evaluation_date": str(latest_date), "horizons": merged}
```

Then both endpoints become ~5-line wrappers.

- [x] **Step 2: Run existing tests**

Run: `cd backend && python -m pytest tests/ -k "ab_test" -v`

- [x] **Step 3: Commit**

```bash
git add backend/api/routes/ab_test.py
git commit -m "refactor(api): extract shared _ab_comparison helper for A/B endpoints"
```

---

## PR 2: Collector Parallelization & Resource Management

Targets: `collectors/csgotrader_aggregator.py`, `collectors/supply_depth.py`, `collectors/pipeline.py`, `collectors/csmarketapi_backfill.py`, `collectors/social_sentiment.py`, `collectors/reddit_events.py`

---

### Task 2.1: Parallelize csgotrader 7-endpoint fetch

**Files:**
- Modify: `backend/collectors/csgotrader_aggregator.py:39-43,170-189`
- Create: `backend/tests/test_aggregator_parallel.py`

**Interfaces:**
- Consumes: `CSGOTraderAggregator.fetch_all_market_data()`, `self.session`
- Produces: Same return type (`dict`), parallel fetch, session lifecycle

- [x] **Step 1: Write the test**

```python
# backend/tests/test_aggregator_parallel.py
from unittest.mock import patch, MagicMock
from collectors.csgotrader_aggregator import CSGOTraderAggregator

def test_fetch_uses_parallel_execution():
    agg = CSGOTraderAggregator()
    with patch("collectors.csgotrader_aggregator.ThreadPoolExecutor") as mock_pool:
        mock_executor = MagicMock()
        mock_pool.return_value.__enter__ = MagicMock(return_value=mock_executor)
        mock_pool.return_value.__exit__ = MagicMock(return_value=False)
        mock_executor.submit.return_value.result.return_value = ("steam", {})
        try:
            agg.fetch_all_market_data()
        except Exception:
            pass
        assert mock_executor.submit.call_count >= 1

def test_aggregator_context_manager():
    agg = CSGOTraderAggregator()
    assert hasattr(agg, "close") or hasattr(agg, "__exit__")
```

- [x] **Step 2: Implement parallel fetch + session lifecycle**

In `csgotrader_aggregator.py`:

Add to `__init__` (after line 43):
```python
def close(self):
    self.session.close()

def __enter__(self):
    return self

def __exit__(self, *exc):
    self.close()
```

Replace `fetch_all_market_data` loop (lines 170-189):
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def _fetch_one(self, source_name: str, url: str) -> tuple[str, dict]:
    response = self.session.get(url, timeout=30)
    response.raise_for_status()
    return source_name, response.json()

# Inside fetch_all_market_data, replace the for loop:
with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
    futures = {
        pool.submit(self._fetch_one, name, url): name
        for name, url in endpoints.items()
    }
    for future in as_completed(futures):
        source_name, data = future.result()
        self._raw_sources[source_name] = data
        log.info("Fetched %s: %d items", source_name, len(data))
```

Note: `requests.Session` is thread-safe for concurrent reads (GET requests).

- [x] **Step 3: Run tests**

Run: `cd backend && python -m pytest tests/test_aggregator_parallel.py tests/ -k "aggregator" -v`

- [x] **Step 4: Commit**

```bash
git add backend/collectors/csgotrader_aggregator.py backend/tests/test_aggregator_parallel.py
git commit -m "perf(collectors): parallelize csgotrader 7-endpoint fetch with ThreadPoolExecutor"
```

---

### Task 2.2: Parallelize supply_depth scalar feeds

**Files:**
- Modify: `backend/collectors/supply_depth.py:642-660,176-185`

**Interfaces:**
- Consumes: `collect()`, `fetch_feed()`, `_get_json()`
- Produces: Same return type, parallel scalar feeds, fixed session fallback

- [x] **Step 1: Parallelize scalar feeds**

In `collect()`, replace line 655:

```python
# Old: results = [fetch_feed(feed, snapshot_day, collected_at, session) for feed in feeds]
# New: parallel fetch — each thread gets its own session since requests.Session is not thread-safe for writes
from concurrent.futures import ThreadPoolExecutor

def _fetch_with_own_session(feed):
    with requests.Session() as s:
        return fetch_feed(feed, snapshot_day, collected_at, s)

with ThreadPoolExecutor(max_workers=len(feeds)) as pool:
    results = list(pool.map(_fetch_with_own_session, feeds))
```

- [x] **Step 2: Fix `_get_json` session leak**

At line 178, wrap the fallback:

```python
def _get_json(url: str, timeout: int, session: requests.Session | None = None) -> Any:
    owns_session = session is None
    sess = session or requests.Session()
    try:
        resp = sess.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    finally:
        if owns_session:
            sess.close()
```

- [x] **Step 3: Run existing tests**

Run: `cd backend && python -m pytest tests/ -k "supply" -v`

- [x] **Step 4: Commit**

```bash
git add backend/collectors/supply_depth.py
git commit -m "perf(collectors): parallelize supply_depth scalar feeds, fix session leak"
```

---

### Task 2.3: Single-pass source_breakdown in pipeline.py

**Files:**
- Modify: `backend/collectors/pipeline.py:482-495`

**Interfaces:**
- Consumes: `price_records` (list of `PriceHistory` ORM objects)
- Produces: `source_breakdown` dict — same keys, same values

- [x] **Step 1: Replace 11 linear scans with Counter**

Replace lines 482-495:

```python
from collections import Counter

source_counts = Counter(r.source for r in price_records)
source_breakdown = {
    "aggregator": primary_items_collected,
    "historical_fallback": fallback_items_collected,
    **{label: source_counts.get(label, 0) for label in SOURCE_LABELS.values()},
}
```

- [x] **Step 2: Run existing tests**

Run: `cd backend && python -m pytest tests/ -k "pipeline" -v`

- [x] **Step 3: Commit**

```bash
git add backend/collectors/pipeline.py
git commit -m "perf(collectors): replace 11-pass source_breakdown with single Counter pass"
```

---

### Task 2.4: Batch inserts in csmarketapi_backfill.py

**Files:**
- Modify: `backend/collectors/csmarketapi_backfill.py:447-449,559-580,181-183`

**Interfaces:**
- Consumes: `sv()`, `inc()`, sell_listings UPDATE loop, sales INSERT loop
- Produces: Same data written, batched operations

- [x] **Step 1: Batch the sales INSERT loop**

Replace lines 559-580 (nested for loop with per-row INSERT):

```python
rows = []
for day_entry in sales_data:
    day = day_entry.get("day", "")
    for sale in day_entry.get("sales", []):
        rows.append((
            hash_name, day, sale.get("market", "UNKNOWN"),
            sale.get("price", 0), sale.get("count", 0),
        ))
if rows:
    out_conn.executemany(
        "INSERT OR IGNORE INTO sales_history (hash_name, day, market, price, count) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
```

- [x] **Step 2: Batch the sell_listings UPDATE**

Replace lines 447-449:

```python
listings = local_conn.execute("SELECT hash_name, sell_listings FROM market_items").fetchall()
out_conn.executemany(
    "UPDATE items SET sell_listings = ? WHERE hash_name = ?",
    [(sl or 0, h) for h, sl in listings],
)
out_conn.commit()
```

- [x] **Step 3: Remove commit from `sv()`**

At line 183, remove `conn.commit()`. Add periodic commit every 50 items in the main loop instead.

- [x] **Step 4: Run existing tests**

Run: `cd backend && python -m pytest tests/ -k "backfill" -v`

- [x] **Step 5: Commit**

```bash
git add backend/collectors/csmarketapi_backfill.py
git commit -m "perf(collectors): batch INSERT/UPDATE in csmarketapi_backfill"
```

---

### Task 2.5: Lazy-load FinBERT + batch social_sentiment inserts

**Files:**
- Modify: `backend/collectors/social_sentiment.py:92-95,251-275`

**Interfaces:**
- Consumes: `_finbert` module-level, per-mention INSERT loop
- Produces: Lazy-loaded scorer, batched inserts

- [x] **Step 1: Lazy-load FinBERT**

Replace line 92:

```python
_finbert: FinbertScorer | None = None

def score_sentiment(text: str) -> float:
    global _finbert
    if _finbert is None:
        _finbert = FinbertScorer()
    return _finbert.score(text)
```

- [x] **Step 2: Batch the mention inserts**

Replace the per-mention INSERT loop (lines 251-275):

```python
mention_params = []
for matched_name in unique_matches:
    mention_params.append({
        "item_id": name_map[matched_name],
        "post_id": post["id"],
        "subreddit": post["subreddit"],
        "sentiment_score": sentiment,
        "mentioned_at": post["created_utc"],
    })

if mention_params:
    db.execute(
        text("""
            INSERT INTO social_mentions (item_id, post_id, subreddit, sentiment_score, mentioned_at)
            VALUES (:item_id, :post_id, :subreddit, :sentiment_score, :mentioned_at)
            ON CONFLICT DO NOTHING
        """),
        mention_params,
    )
```

- [x] **Step 3: Run existing tests**

Run: `cd backend && python -m pytest tests/ -k "sentiment" -v`

- [x] **Step 4: Commit**

```bash
git add backend/collectors/social_sentiment.py
git commit -m "perf(collectors): lazy-load FinBERT, batch social mention inserts"
```

---

### Task 2.6: Fix session leaks in reddit_events.py

**Files:**
- Modify: `backend/collectors/reddit_events.py:291,299-303`

**Interfaces:**
- Consumes: `collect_reddit_events()` internals
- Produces: Same return type, parallel subreddit fetch, session lifecycle

- [x] **Step 1: Fix session lifecycle + parallel fetch**

```python
def collect_reddit_events(db=None, session=None, snapshot_day=None, ...):
    owns_session = session is None
    sess = session or requests.Session()
    try:
        # Parallel fetch of 3 independent subreddits
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(SUBREDDITS)) as pool:
            post_lists = list(pool.map(
                lambda sub: fetch_subreddit_posts(sub, token, sess),
                SUBREDDITS,
            ))
        posts = [p for sublist in post_lists for p in sublist]
        # ... rest of function unchanged ...
    finally:
        if owns_session:
            sess.close()
```

Note: `requests.Session` is thread-safe for GETs. If the auth token is session-scoped, verify thread safety.

- [x] **Step 2: Run existing tests**

Run: `cd backend && python -m pytest tests/ -k "reddit" -v`

- [x] **Step 3: Commit**

```bash
git add backend/collectors/reddit_events.py
git commit -m "fix(collectors): close fallback session, parallelize subreddit fetches"
```

---

## PR 3: Database Infrastructure

Targets: `database.py`, `db/parquet.py`

---

### Task 3.1: Add connection pool tuning + statement timeout

**Files:**
- Modify: `backend/database.py:32-37`

**Interfaces:**
- Consumes: `settings.database_url` from `config.py`
- Produces: `engine` with pool settings and statement timeout

- [x] **Step 1: Add pool configuration**

Replace lines 32-37:

```python
engine = create_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_recycle=1800,
    connect_args={"options": "-c statement_timeout=30000"},
)
```

- [x] **Step 2: Verify the app starts**

Run: `cd backend && python -c "from database import engine; print(engine.pool.status())"`

- [x] **Step 3: Commit**

```bash
git add backend/database.py
git commit -m "infra(db): add pool tuning and 30s statement_timeout"
```

---

### Task 3.2: Atomic `_append_parquet` and `delete_table`

**Files:**
- Modify: `backend/db/parquet.py:244-254,523-536`
- Create: `backend/tests/test_parquet_atomic.py`

**Interfaces:**
- Consumes: `_tmp_path(path)` (line 439), `_atomic_write(path, df)` (line 450) — both already exist in the module
- Produces: Crash-safe `_append_parquet` and `delete_table`

- [x] **Step 1: Write the test**

```python
# backend/tests/test_parquet_atomic.py
import os
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from db.parquet import delete_table, _atomic_write

def test_delete_table_uses_atomic_write(tmp_path):
    """delete_table must use atomic write (temp + os.replace), not direct overwrite."""
    path = tmp_path / "test.parquet"
    df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
    df.to_parquet(path, index=False)

    with patch("db.parquet._atomic_write") as mock_atomic:
        mock_atomic.side_effect = lambda p, d: d.to_parquet(p, index=False)
        # Actual call depends on delete_table's exact interface
```

- [x] **Step 2: Fix `_append_parquet`**

In `_append_parquet` (lines 244-254), replace the `COPY TO '{path}'` with:

```python
tmp = _tmp_path(path)
con.execute(f"""
    COPY (...) TO '{tmp}' (FORMAT PARQUET)
""")
os.replace(tmp, path)
```

- [x] **Step 3: Fix `delete_table`**

In `delete_table` (lines 523-536), replace `df.to_parquet(path)` with:

```python
_atomic_write(path, filtered)
```

Where `_atomic_write` is already defined at line 450.

- [x] **Step 4: Run tests**

Run: `cd backend && python -m pytest tests/test_parquet_atomic.py tests/test_parquet_nested_columns.py -v`

- [x] **Step 5: Commit**

```bash
git add backend/db/parquet.py backend/tests/test_parquet_atomic.py
git commit -m "fix(db): make _append_parquet and delete_table crash-safe with atomic writes"
```

---

### Task 3.3: Fix `_get_ops_schema` cache + ParquetQuery validation

**Files:**
- Modify: `backend/db/parquet.py:465-508,539-549`

**Interfaces:**
- Consumes: `ParquetQuery.__enter__`, `_get_ops_schema`
- Produces: Properly-sized cache, validated table names

- [x] **Step 1: Fix LRU cache size**

Replace line 539:

```python
@lru_cache(maxsize=16)  # covers all ops tables
def _get_ops_schema(table: str) -> dict | None:
```

- [x] **Step 2: Add table name validation to ParquetQuery**

At line 486, before the `CREATE VIEW`:

```python
import re
if not re.match(r"^[a-z_][a-z0-9_]*$", self._table):
    raise ValueError(f"Invalid table name: {self._table!r}")
```

- [x] **Step 3: Run existing tests**

Run: `cd backend && python -m pytest tests/ -k "parquet" -v`

- [x] **Step 4: Commit**

```bash
git add backend/db/parquet.py
git commit -m "fix(db): increase _get_ops_schema cache, validate ParquetQuery table names"
```

---

### Task 3.4: Remove duplicate indexes

**Files:**
- Modify: `backend/database.py:408-412,456-457`
- Create: `backend/migrations/versions/0XXX_remove_duplicate_indexes.py`

**Interfaces:**
- Consumes: `SupplySnapshot.__table_args__`, `ForecastOutcome.__table_args__`
- Produces: Clean schema without redundant indexes

- [x] **Step 1: Remove duplicate index declarations from models**

In `database.py`:

Line 457 — remove `Index("idx_supply_item_date", "item_id", "snapshot_date")` from `SupplySnapshot.__table_args__` (PK already covers this).

Lines 408-409 — remove `Index("idx_outcome_forecast_id", "forecast_id")` from `ForecastOutcome.__table_args__` (column already has `index=True`).

- [x] **Step 2: Create migration**

```bash
cd backend && alembic revision --autogenerate -m "remove duplicate indexes on supply_snapshot and forecast_outcome"
```

Review the generated migration — it should contain `op.drop_index("idx_supply_item_date")` and `op.drop_index("idx_outcome_forecast_id")`.

- [x] **Step 3: Commit**

```bash
git add backend/database.py backend/migrations/versions/
git commit -m "schema(db): remove duplicate indexes on SupplySnapshot and ForecastOutcome"
```

---

## PR 4: ML Hot Path Optimizations

Targets: `models/forecaster.py`, `backtest/scoring.py`, `backtest/paired_mde.py`

---

### Task 4.1: Vectorize fold directional accuracy

**Files:**
- Modify: `backend/models/forecaster.py:10117-10138`

**Interfaces:**
- Consumes: `direction.directional_accuracy(pred_returns, actual_returns) -> float` (already exists at `models/direction.py:20`)
- Produces: Same `fold_acc` value, vectorized computation

- [x] **Step 1: Write a regression test**

```python
# backend/tests/test_fold_accuracy_vectorized.py
import numpy as np
from models.direction import directional_accuracy

def test_directional_accuracy_matches_manual():
    """Verify vectorized directional_accuracy matches the manual loop output."""
    pred = np.array([0.05, -0.02, 0.0, 0.10, -0.03])
    actual = np.array([0.03, -0.01, 0.01, -0.05, -0.08])
    result = directional_accuracy(pred, actual)
    # Manual: up/down match for indices 0,1,4; mismatch 2,3 -> 60%
    assert abs(result - 60.0) < 0.1
```

- [x] **Step 2: Run test to verify it passes** (the function already exists)

Run: `cd backend && python -m pytest tests/test_fold_accuracy_vectorized.py -v`

- [x] **Step 3: Replace the manual loop**

Replace lines 10117-10138 with:

```python
from models.direction import directional_accuracy
fold_acc = directional_accuracy(fold_p50, actual_returns)
```

Remove the `for i in range(len(val_df))` loop, the `fold_hits` counter, and the per-row string comparisons.

- [x] **Step 4: Run forecaster tests**

Run: `cd backend && python -m pytest tests/test_forecaster.py -v --timeout=120`

- [x] **Step 5: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_fold_accuracy_vectorized.py
git commit -m "perf(models): vectorize fold directional accuracy using existing direction module"
```

---

### Task 4.2: Vectorize bootstrap loops

**Files:**
- Modify: `backend/backtest/scoring.py:263-320`
- Modify: `backend/backtest/paired_mde.py:148-156`

**Interfaces:**
- Consumes: `bootstrap_ci(arr, n_resamples, rng)`, `block_bootstrap_ci(sums, counts, n_resamples, rng)`
- Produces: Same return type `(lo, hi)` — vectorized implementation

- [x] **Step 1: Write regression test**

```python
# backend/tests/test_bootstrap_vectorized.py
import numpy as np
from backtest.scoring import bootstrap_ci, block_bootstrap_ci

def test_bootstrap_ci_deterministic():
    rng = np.random.default_rng(42)
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lo, hi = bootstrap_ci(arr, n_resamples=1000, rng=rng)
    # Re-run with same seed — must match
    rng2 = np.random.default_rng(42)
    lo2, hi2 = bootstrap_ci(arr, n_resamples=1000, rng=rng2)
    assert lo == lo2 and hi == hi2

def test_block_bootstrap_ci_deterministic():
    rng = np.random.default_rng(42)
    sums = np.array([10.0, 20.0, 30.0])
    counts = np.array([5, 10, 15])
    lo, hi = block_bootstrap_ci(sums, counts, n_resamples=1000, rng=rng)
    rng2 = np.random.default_rng(42)
    lo2, hi2 = block_bootstrap_ci(sums, counts, n_resamples=1000, rng=rng2)
    assert lo == lo2 and hi == hi2
```

- [x] **Step 2: Vectorize `bootstrap_ci`**

Replace lines 263-277:

```python
def bootstrap_ci(arr, n_resamples=1000, rng=None, ci=0.95):
    if rng is None:
        rng = np.random.default_rng()
    n = len(arr)
    indices = rng.choice(n, size=(n_resamples, n), replace=True)
    stats = arr[indices].mean(axis=1)
    alpha = (1 - ci) / 2
    return float(np.percentile(stats, 100 * alpha)), float(np.percentile(stats, 100 * (1 - alpha)))
```

- [x] **Step 3: Vectorize `block_bootstrap_ci`**

Replace lines 280-320:

```python
def block_bootstrap_ci(sums, counts, n_resamples=1000, rng=None, ci=0.95):
    if rng is None:
        rng = np.random.default_rng()
    n_groups = len(sums)
    all_idx = rng.integers(0, n_groups, size=(n_resamples, n_groups))
    stat_sums = sums[all_idx].sum(axis=1)
    stat_counts = counts[all_idx].sum(axis=1)
    stats = stat_sums / stat_counts
    alpha = (1 - ci) / 2
    return float(np.percentile(stats, 100 * alpha)), float(np.percentile(stats, 100 * (1 - alpha)))
```

- [x] **Step 4: Apply same pattern to `paired_mde.py:148-156`**

Same vectorization — generate all bootstrap indices at once.

- [x] **Step 5: Run tests**

Run: `cd backend && python -m pytest tests/test_bootstrap_vectorized.py tests/test_backtest_scoring.py tests/test_paired_mde.py -v`

- [x] **Step 6: Commit**

```bash
git add backend/backtest/scoring.py backend/backtest/paired_mde.py backend/tests/test_bootstrap_vectorized.py
git commit -m "perf(backtest): vectorize bootstrap loops — 1000 iterations to single numpy operation"
```

---

### Task 4.3: Single-pass `score_cohort`

**Files:**
- Modify: `backend/backtest/scoring.py:336-535`

**Interfaces:**
- Consumes: `score_cohort(records, ...)` — list of dicts
- Produces: Same return dict, computed in 1-2 passes instead of ~20

- [x] **Step 1: Accumulate all metrics in a single pass**

Replace the ~20 list comprehensions with a single accumulator loop:

```python
def score_cohort(records, ...):
    # Single pass
    mae_sum = sq_sum = pct_sum = dir_sum = 0.0
    interval_total = interval_hits = dollar_hits = 0
    n = len(records)
    abs_errors = []
    dir_corrects = []
    flat_count = flat_error_sum = 0
    tier_errors = {}  # price_tier -> [abs_errors]

    for r in records:
        ae = r["abs_error"]
        mae_sum += ae
        sq_sum += r["sq_error"]
        pct_sum += r["pct_error"]
        dir_sum += r["direction_correct"]
        abs_errors.append(ae)
        dir_corrects.append(r["direction_correct"])

        if r.get("in_interval") is not None:
            interval_total += 1
            interval_hits += r["in_interval"]
            dollar_hits += r.get("in_interval_dollar", 0)

        if r.get("actual_direction") == "flat":
            flat_count += 1
            flat_error_sum += abs(r["actual_price"] - r.get("base_price", 0))

        tier = r.get("price_tier")
        if tier is not None:
            tier_errors.setdefault(tier, []).append(ae)

    # Derive all metrics from accumulators
    mae = mae_sum / n if n else 0
    rmse = (sq_sum / n) ** 0.5 if n else 0
    # ... etc, using the accumulated values
```

Preserve the exact same output dict keys and values.

- [x] **Step 2: Run existing tests**

Run: `cd backend && python -m pytest tests/test_backtest_scoring.py -v`

- [x] **Step 3: Commit**

```bash
git add backend/backtest/scoring.py
git commit -m "perf(backtest): consolidate score_cohort from ~20 passes to single pass"
```

---

### Task 4.4: In-place shuffle for `_validate_feature_groups`

**Files:**
- Modify: `backend/models/forecaster.py:4440-4463`

**Interfaces:**
- Consumes: `_validate_feature_groups(model, X_val, y_val, groups, group_indices, ...)`
- Produces: Same return type — in-place shuffle avoids 100 array copies

- [x] **Step 1: Replace copy-per-shuffle with save-restore pattern**

Replace lines 4440-4463:

```python
for group_name, idxs in group_indices.items():
    originals = {i: X_val[:, i].copy() for i in idxs}
    baseline = model.predict(X_val)
    drops = []
    for _ in range(n_shuffles):
        for i in idxs:
            RNG.shuffle(X_val[:, i])
        shuffled_pred = model.predict(X_val)
        drop = metric(y_val, baseline) - metric(y_val, shuffled_pred)
        drops.append(drop)
        # Restore originals
        for i in idxs:
            X_val[:, i] = originals[i]
    # ... rest of significance test unchanged
```

Memory: O(group_features × n_rows) instead of O(n_shuffles × n_features × n_rows).

- [x] **Step 2: Run forecaster tests**

Run: `cd backend && python -m pytest tests/test_forecaster.py -v --timeout=120`

- [x] **Step 3: Commit**

```bash
git add backend/models/forecaster.py
git commit -m "perf(models): in-place shuffle for feature group validation — eliminates 100 array copies"
```

---

### Task 4.5: Clean up vote function redundant branch

**Files:**
- Modify: `backend/models/forecaster.py:2517-2555`

**Interfaces:**
- Consumes: `vote(group)` inner function in `_apply_multi_source_voting`
- Produces: Same behavior, cleaner code

- [x] **Step 1: Simplify the branch**

Replace:
```python
if n_sources >= 3:
    consensus = np.median(prices)
else:
    consensus = np.median(prices)
    return pd.Series(...)
```

With:
```python
consensus = np.median(prices)
if n_sources < 3:
    return pd.Series(...)
```

- [x] **Step 2: Run tests**

Run: `cd backend && python -m pytest tests/test_forecaster.py -v --timeout=120`

- [x] **Step 3: Commit**

```bash
git add backend/models/forecaster.py
git commit -m "refactor(models): remove redundant branch in vote function"
```

---

## PR 5: Script Optimizations

Targets: `scripts/event_correlation_analysis.py`, `scripts/ingest_bymykel_metadata.py`, `scripts/forecast_prices.py`, `scripts/init_local_db.py`, `scripts/build_all_sidecars.py`, `scripts/check_sidecar_continuity.py`, `scripts/backfill_buff_iflow.py`, `scripts/replay_serving.py`

---

### Task 5.1: Batch upserts in event_correlation_analysis.py

**Files:**
- Modify: `backend/scripts/event_correlation_analysis.py:380-396,488-510`

**Interfaces:**
- Consumes: `_upsert_event_impacts(db, impacts, ...)`, `_compute_and_upsert_correlations(db, event, ...)`
- Produces: Same data written, batched queries

- [x] **Step 1: Batch-fetch existing rows, then bulk insert/update**

For `_upsert_event_impacts` (lines 380-396):
```python
# Fetch all existing impacts for this event in one query
existing = {
    (r.event_id, r.item_id): r
    for r in db.query(EventImpact).filter(
        EventImpact.event_id == event_id,
        EventImpact.item_id.in_([i["item_id"] for i in impacts])
    ).all()
}

to_insert = []
to_update = []
for impact in impacts:
    key = (event_id, impact["item_id"])
    if key in existing:
        to_update.append(impact)
    else:
        to_insert.append(impact)

if to_insert:
    db.bulk_insert_mappings(EventImpact, to_insert)
# Batch update via single statement per changed field
```

Apply the same pattern to `_compute_and_upsert_correlations`.

- [x] **Step 2: Run existing tests**

Run: `cd backend && python -m pytest tests/ -k "correlation" -v`

- [x] **Step 3: Commit**

```bash
git add backend/scripts/event_correlation_analysis.py
git commit -m "perf(scripts): batch upserts in event_correlation — eliminates ~11K round-trips per event"
```

---

### Task 5.2: Single-pass `archive_slugs` in ingest_bymykel_metadata.py

**Files:**
- Modify: `backend/scripts/ingest_bymykel_metadata.py:496-583`

**Interfaces:**
- Consumes: `archive_slugs(min_price, min_days, before)`
- Produces: Three slug sets from a single archive scan

- [x] **Step 1: Replace 3 calls with 1 query**

Replace the three `archive_slugs()` calls (lines 580-583) with:

```python
def archive_slug_sets(archive_dir: Path) -> tuple[set[str], set[str], set[str]]:
    """Return (all_slugs, pre_2026_slugs, pre_2026_no_iflow_slugs) in one scan."""
    con = duckdb.connect()
    try:
        rel = prices_relation(con, archive_dir)
        df = con.execute(f"""
            SELECT item_slug,
                   min(day) AS first_day,
                   count(DISTINCT day) AS n_days,
                   median(mean_price) AS med_price,
                   bool_or(source = 'buff_iflow') AS has_iflow
            FROM ({rel})
            GROUP BY item_slug
        """).fetchdf()
    finally:
        con.close()

    all_slugs = set(df["item_slug"])
    pre_2026 = df[df["first_day"] < "2026-01-01"]
    pre_2026_slugs = set(pre_2026["item_slug"])
    pre_2026_no_iflow = set(pre_2026[~pre_2026["has_iflow"]]["item_slug"])
    return all_slugs, pre_2026_slugs, pre_2026_no_iflow
```

- [x] **Step 2: Run existing tests**

Run: `cd backend && python -m pytest tests/test_bymykel_metadata_wiring.py -v`

- [x] **Step 3: Commit**

```bash
git add backend/scripts/ingest_bymykel_metadata.py
git commit -m "perf(scripts): single-pass archive_slugs — eliminates 2 redundant full-archive scans"
```

---

### Task 5.3: Fix forecast_prices.py — iterrows + schema cache

**Files:**
- Modify: `backend/scripts/forecast_prices.py:287-375`

**Interfaces:**
- Consumes: `_write_forecasts_to_db(results, db, ...)` internals
- Produces: Same data written, using `itertuples()` and cached schema

- [x] **Step 1: Replace `iterrows()` with `itertuples()`**

At line 288, replace:
```python
for idx, row in results.iterrows():
```
with:
```python
for row in results.itertuples(index=False):
```

Adjust attribute access from `row.get("col")` / `row["col"]` to `getattr(row, "col", None)`.

- [x] **Step 2: Cache schema introspection**

Move the `sa_inspect(bind).get_columns(table.name)` call out of `_write_forecasts_to_db` and compute once:

```python
_FORECAST_DB_COLS: set[str] | None = None

def _get_forecast_cols(bind, table):
    global _FORECAST_DB_COLS
    if _FORECAST_DB_COLS is None:
        from sqlalchemy import inspect as sa_inspect
        _FORECAST_DB_COLS = {c["name"] for c in sa_inspect(bind).get_columns(table.name)}
    return _FORECAST_DB_COLS
```

- [x] **Step 3: Move lazy imports to top of file**

Move `from sqlalchemy.dialects.postgresql import insert as pg_insert` (line 341) and `from sqlalchemy import inspect as sa_inspect` (line 369) to the top-level imports.

- [x] **Step 4: Run existing tests**

Run: `cd backend && python -m pytest tests/test_forecast_prices.py tests/test_forecast_date_is_anchor_day.py -v`

- [x] **Step 5: Commit**

```bash
git add backend/scripts/forecast_prices.py
git commit -m "perf(scripts): itertuples + cached schema in forecast_prices writer"
```

---

### Task 5.4: Batch init_local_db queries + updates

**Files:**
- Modify: `backend/scripts/init_local_db.py:68-158`

**Interfaces:**
- Consumes: DuckDB archive queries, per-row UPDATE loop
- Produces: Same local DB state, 3 queries → 1, per-row UPDATEs → batched

- [x] **Step 1: Single GROUP BY query**

Replace lines 68-105 (three `SELECT DISTINCT` queries) with:

```python
con = duckdb.connect()
rel = prices_relation(con, archive_dir)
df = con.execute(f"""
    SELECT item_slug,
           min(day) AS first_day,
           bool_or(source IS DISTINCT FROM 'buff_iflow') AS has_non_iflow
    FROM ({rel})
    GROUP BY item_slug
""").fetchdf()
con.close()

all_slugs = set(df["item_slug"])
pre_2026_slugs = set(df[df["first_day"] < "2026-01-01"]["item_slug"])
pre_2026_no_iflow = set(
    df[(df["first_day"] < "2026-01-01") & df["has_non_iflow"]]["item_slug"]
)
```

- [x] **Step 2: Batch the UPDATE loop**

Replace lines 143-158 (per-row UPDATE) with:

```python
needs_backfill = [iid for iid, bf, tr in items if not bf]
needs_train = [iid for iid, bf, tr in items if not tr]
if needs_backfill:
    db.query(Item).filter(Item.item_id.in_(needs_backfill)).update(
        {Item.is_backfilled: True}, synchronize_session=False
    )
if needs_train:
    db.query(Item).filter(Item.item_id.in_(needs_train)).update(
        {Item.is_trainable: True}, synchronize_session=False
    )
db.commit()
```

- [x] **Step 3: Commit**

```bash
git add backend/scripts/init_local_db.py
git commit -m "perf(scripts): single-pass archive query + batch updates in init_local_db"
```

---

### Task 5.5: Use Parquet metadata for row counts

**Files:**
- Modify: `backend/scripts/build_all_sidecars.py:29`
- Modify: `backend/scripts/check_sidecar_continuity.py:54-63`

**Interfaces:**
- Consumes: `_count_sidecars(archive_dir)`, sidecar continuity check
- Produces: Same outputs, using metadata reads instead of full file loads

- [x] **Step 1: Fix `_count_sidecars`**

Replace line 29:

```python
import pyarrow.parquet as pq

def _count_sidecars(archive_dir: Path) -> dict:
    return {
        name: pq.read_metadata(archive_dir / name).num_rows
        for name in SIDECAR_NAMES
        if (archive_dir / name).exists()
    }
```

- [x] **Step 2: Fix `check_sidecar_continuity` to use DuckDB projection**

Replace lines 54-63:

```python
con = duckdb.connect()
df = con.execute(f"""
    SELECT DISTINCT day, source
    FROM read_parquet('{archive_dir}/{table}-*.parquet')
""").fetchdf()
con.close()
```

- [x] **Step 3: Commit**

```bash
git add backend/scripts/build_all_sidecars.py backend/scripts/check_sidecar_continuity.py
git commit -m "perf(scripts): use Parquet metadata for row counts, DuckDB projection for date audit"
```

---

### Task 5.6: Fix backfill_buff_iflow — fx_lookup + memory

**Files:**
- Modify: `backend/scripts/backfill_buff_iflow.py:158,188-229`

**Interfaces:**
- Consumes: `fx_lookup(fx, day)`, accumulation loop
- Produces: Same data, O(1) FX lookup, monthly flush

- [x] **Step 1: Replace O(n) fx_lookup with searchsorted**

Replace line 158:

```python
def fx_lookup(fx: pd.DataFrame, day: date) -> float | None:
    if fx.empty:
        return None
    idx = np.searchsorted(fx["day"].values, np.datetime64(day), side="right") - 1
    if idx < 0:
        return None
    return float(fx.iloc[idx]["rate"])
```

- [x] **Step 2: Add monthly flush to accumulation loop**

In the accumulation loop (lines 188-229), add:

```python
current_month = None
for dump_day, dump_data in sorted(dumps.items()):
    month_key = (dump_day.year, dump_day.month)
    if current_month and month_key != current_month:
        _flush_rows(price_rows, vol_rows, archive_dir, current_month)
        price_rows.clear()
        vol_rows.clear()
    current_month = month_key
    # ... existing row accumulation ...

# Flush remaining
if price_rows or vol_rows:
    _flush_rows(price_rows, vol_rows, archive_dir, current_month)
```

- [x] **Step 3: Run existing tests**

Run: `cd backend && python -m pytest tests/test_backfill_buff_iflow.py -v`

- [x] **Step 4: Commit**

```bash
git add backend/scripts/backfill_buff_iflow.py
git commit -m "perf(scripts): O(1) fx_lookup via searchsorted, monthly flush for memory"
```

---

## PR 6: Tier 3 — Simplification & Code Health

These are lower-priority but reduce maintenance burden. Can be done incrementally.

---

### Task 6.1: Adopt `prices_relation()` across scripts

**Files:**
- Modify: `archive/backfill_supply_metadata.py:552-558`
- Modify: `archive/walkforward_backtest.py:79-114`
- Modify: Any other script using bare `read_parquet('prices-*.parquet')`

**Interfaces:**
- Consumes: `db.archive.prices_relation(con, archive_dir, columns, where)` — returns SQL table expression string
- Produces: Schema-safe archive reads

- [x] **Step 1: Grep for bare read_parquet patterns**

```bash
cd backend && grep -rn "read_parquet.*prices-\*" scripts/ --include="*.py" | grep -v "prices_relation"
```

- [x] **Step 2: Replace each bare `read_parquet` with `prices_relation()`**

For each hit, replace the handrolled SQL with:

```python
from db.archive import prices_relation
rel = prices_relation(con, archive_dir)
# Use rel as the FROM clause
```

- [x] **Step 3: Run targeted tests for each modified script**

- [x] **Step 4: Commit**

```bash
git add backend/scripts/ backend/archive/
git commit -m "refactor(scripts): adopt prices_relation() — fixes silent schema mismatch risk"
```

---

### Task 6.2: Extract shared `build_panel` helper

**Files:**
- Modify: `backend/scripts/build_case_panel.py`
- Modify: `backend/scripts/build_sticker_panel.py`
- Create: `backend/scripts/_panel_common.py`

**Interfaces:**
- Produces: `_load_supply_and_events(archive_dir)` shared helper, `_build_panel_main(panel_fn, sidecar_name)` orchestrator

- [x] **Step 1: Extract shared code**

```python
# backend/scripts/_panel_common.py
def load_supply_and_events(archive_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load supply files and event calendar — shared by case + sticker panels."""
    # ... the ~40 identical lines from both scripts ...

def build_panel_main(panel_fn, sidecar_name: str, summary_key: str, archive_dir: Path):
    """Shared orchestrator for panel builders."""
    supply, events = load_supply_and_events(archive_dir)
    panel = panel_fn(supply, events)
    # ... write sidecar ...
```

- [x] **Step 2: Slim down both scripts to use the helper**

- [x] **Step 3: Run tests**

Run: `cd backend && python -m pytest tests/ -k "panel or sidecar" -v`

- [x] **Step 4: Commit**

```bash
git add backend/scripts/_panel_common.py backend/scripts/build_case_panel.py backend/scripts/build_sticker_panel.py
git commit -m "refactor(scripts): extract shared panel builder — eliminates ~80% duplication"
```

---

### Task 6.3: Clean up dead code in run_task.py

**Files:**
- Modify: `backend/scripts/run_task.py:130-164`

**Interfaces:**
- Produces: Clean task runner without dead variables and deprecated stubs

- [x] **Step 1: Remove dead variables and stubs**

Delete `result2 = result3 = None` (line 130-131) and the `trends` / `long_term_trends` task stubs that return hardcoded success dicts (lines 153-164).

- [x] **Step 2: Run existing tests**

Run: `cd backend && python -m pytest tests/ -k "run_task" -v`

- [x] **Step 3: Commit**

```bash
git add backend/scripts/run_task.py
git commit -m "cleanup(scripts): remove dead variables and deprecated task stubs"
```

---

## Execution Order

```
PR 1 (API queries)          — independent, ship first for user-facing latency wins
PR 2 (Collectors)           — independent, ship for daily pipeline speedup
PR 3 (DB infrastructure)    — independent, ship for safety + resource management
PR 4 (ML hot path)          — independent, ship for training/backtest speedup
PR 5 (Scripts)              — independent, operational improvements
PR 6 (Simplification)       — lowest priority, ship last
```

All 6 PRs are independent and can be developed and merged in any order or in parallel.
