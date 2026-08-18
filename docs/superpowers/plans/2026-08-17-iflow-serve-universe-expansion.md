# iflow Serve-Universe Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve forecasts for ~11,600 iflow-backed ≥$1 items without growing the ~926-item training set.

**Architecture:** `items.is_backfilled` (any pre-2026 archive row) already gates BOTH training and serving; it widens for free once iflow is ingested. Add a narrow `items.is_trainable` flag (pre-2026 row from a **non-`buff_iflow`** source) that the training path reads instead, so serving widens while training stays put across retrains. No retrain; the existing artifact serves the enlarged frame.

**Tech Stack:** Python 3.13 (3.11 in CI), SQLAlchemy, DuckDB over Parquet, LightGBM, pytest. Run from `backend/` via `venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-08-17-iflow-serve-universe-expansion-design.md`

## Global Constraints

- Run all commands from `backend/` through `venv/bin/python` (venv wrappers may be stale; call `venv/bin/python -m …`).
- `backend/.env` points at **production** Supabase. NEVER run a write-path script from `backend/` without overriding `DATABASE_URL` to a local sqlite first. Tests use their own sessions/fixtures — verify each new test targets sqlite, not prod.
- The training universe invariant: `is_trainable ∩ median-price ≥ $1` must stay ≈ 926 items. This is the property the whole change protects.
- `buff_iflow` is an ordinary ask source — do NOT add it to `BID_SOURCES` / `TRAILING_WINDOW_SOURCES` / `STEAM_SPOT_SOURCES`. It is allowed to vote.
- Follow existing repo test style: `backend/tests/test_<name>.py`, run with `venv/bin/python -m pytest tests/test_<name>.py -q`.

---

### Task 1: Add `is_trainable` column and `trainable_item_clause()`

**Files:**
- Modify: `backend/database.py:54` (Item model), `backend/database.py:98-105` (clause helpers)
- Test: `backend/tests/test_trainable_flag.py`

**Interfaces:**
- Produces: `Item.is_trainable` (Integer, default 0); `trainable_item_clause() -> Item.is_trainable == 1`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_trainable_flag.py
from database import Item, trainable_item_clause, backfilled_item_clause

def test_item_has_is_trainable_column():
    assert "is_trainable" in Item.__table__.columns
    assert Item.__table__.columns["is_trainable"].default.arg == 0

def test_trainable_clause_distinct_from_backfilled():
    assert str(trainable_item_clause()) == "items.is_trainable = :is_trainable_1"
    assert str(backfilled_item_clause()) == "items.is_backfilled = :is_backfilled_1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_trainable_flag.py -q`
Expected: FAIL (`ImportError: cannot import name 'trainable_item_clause'`).

- [ ] **Step 3: Implement**

In `backend/database.py`, after the `is_backfilled` column (line 54) add:

```python
    is_trainable = Column(Integer, default=0)  # boolean: eligible for the TRAIN universe (non-iflow pre-2026 history)
```

After `backfilled_item_clause()` (line 105) add:

```python
def trainable_item_clause():
    """Items eligible for the TRAINING universe: pre-2026 history from a
    non-`buff_iflow` source. Distinct from `backfilled_item_clause()`, which is
    the wider SERVE universe that includes iflow-backfilled items."""
    return Item.is_trainable == 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_trainable_flag.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/database.py backend/tests/test_trainable_flag.py
git commit -m "feat: add items.is_trainable flag and trainable_item_clause"
```

---

### Task 2: Derive `is_trainable` in `init_local_db.py` (source-aware)

**Files:**
- Modify: `backend/scripts/init_local_db.py:84-134`
- Test: `backend/tests/test_init_trainable_derivation.py`

**Interfaces:**
- Consumes: `Item.is_trainable` (Task 1).
- Produces: after `init_local_db` runs, an item is `is_trainable=1` iff it has a pre-2026 archive row whose `source IS DISTINCT FROM 'buff_iflow'`; `is_backfilled` unchanged (any pre-2026 row).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_init_trainable_derivation.py
import duckdb, pandas as pd
from pathlib import Path

def _write_archive(tmp_path):
    # non-iflow item: pre-2026 NULL-source row  -> backfilled AND trainable
    # iflow-only item: pre-2026 buff_iflow row  -> backfilled, NOT trainable
    df = pd.DataFrame([
        {"item_slug": "AK-47 | Redline (Field-Tested)", "day": "2024-06-01", "source": None,        "mean_price": 12.0, "volume": None, "ingested_at": "2026-01-01"},
        {"item_slug": "AWP | Acheron (Field-Tested)",   "day": "2024-06-01", "source": "buff_iflow", "mean_price": 3.0,  "volume": None, "ingested_at": "2026-01-01"},
    ])
    p = tmp_path / "prices-fixture-2024-06.parquet"
    df.to_parquet(p)
    return tmp_path

def test_iflow_only_item_is_backfilled_but_not_trainable(tmp_path):
    arc = _write_archive(tmp_path)
    con = duckdb.connect()
    backfilled = {r[0] for r in con.sql(
        f"SELECT DISTINCT item_slug FROM read_parquet('{arc}/prices-*.parquet', union_by_name=true) WHERE day < '2026-01-01'").fetchall()}
    trainable = {r[0] for r in con.sql(
        f"SELECT DISTINCT item_slug FROM read_parquet('{arc}/prices-*.parquet', union_by_name=true) WHERE day < '2026-01-01' AND source IS DISTINCT FROM 'buff_iflow'").fetchall()}
    con.close()
    assert "AWP | Acheron (Field-Tested)" in backfilled
    assert "AWP | Acheron (Field-Tested)" not in trainable   # iflow-only -> serve but not train
    assert "AK-47 | Redline (Field-Tested)" in trainable       # non-iflow -> train
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_init_trainable_derivation.py -q`
Expected: FAIL (no code yet asserts the semantics — this test locks the SQL predicate before wiring it into the script). If it passes on the raw SQL alone, proceed to Step 3 to wire the predicate into `init_local_db.py` and add the DB-write assertions there.

- [ ] **Step 3: Implement**

In `backend/scripts/init_local_db.py`, alongside the `backfilled` set (line ~85), add a `trainable` set:

```python
        trainable = {
            r[0] for r in con.sql(f"""
                SELECT DISTINCT item_slug
                FROM read_parquet('{ARCHIVE_DIR}/prices-*.parquet', union_by_name=true)
                WHERE day < '2026-01-01' AND source IS DISTINCT FROM 'buff_iflow'
            """).fetchall()
        }
```

In the insert loop (line ~106) set `is_trainable=1 if slug in trainable else 0`. In the re-derive loop (line ~124), extend it to also correct `is_trainable`:

```python
    for item_id, bf, tr in db.query(Item.item_id, Item.is_backfilled, Item.is_trainable).all():
        want_bf = 1 if item_id in backfilled else 0
        want_tr = 1 if item_id in trainable else 0
        if (bf or 0) != want_bf or (tr or 0) != want_tr:
            db.query(Item).filter(Item.item_id == item_id).update(
                {"is_backfilled": want_bf, "is_trainable": want_tr,
                 "updated_at": datetime.now(timezone.utc).replace(tzinfo=None)})
            changed += 1
```

Add `is_trainable` to the summary log line.

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_init_trainable_derivation.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/init_local_db.py backend/tests/test_init_trainable_derivation.py
git commit -m "feat: derive is_trainable (non-iflow pre-2026 history) in init_local_db"
```

---

### Task 3: Route training to `is_trainable`, serving to `is_backfilled` via a `universe` param

**Files:**
- Modify: `backend/models/forecaster.py:1787-1810` (`_resolve_backfilled_slugs`), `:1734-1745` (`fetch_price_history`), `:5049-5070` (`_voted_cache_key`), `:4912-4917` (`build_training_data`), `:5315` (`train` call).
- Test: `backend/tests/test_universe_routing.py`

**Interfaces:**
- Consumes: `Item.is_backfilled`, `Item.is_trainable`.
- Produces: `_resolve_backfilled_slugs(universe="serve")` reads `is_backfilled`; `universe="train"` reads `is_trainable`. `fetch_price_history(..., universe="serve")` and `build_training_data(..., universe="serve")` thread it through. `_voted_cache_key` includes `universe` so train/serve frames cache separately.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_universe_routing.py
import inspect
from models.forecaster import ItemForecaster

def test_resolve_slugs_accepts_universe():
    sig = inspect.signature(ItemForecaster._resolve_backfilled_slugs)
    assert "universe" in sig.parameters

def test_fetch_and_build_thread_universe():
    assert "universe" in inspect.signature(ItemForecaster.fetch_price_history).parameters
    assert "universe" in inspect.signature(ItemForecaster.build_training_data).parameters

def test_universe_selects_column():
    src = inspect.getsource(ItemForecaster._resolve_backfilled_slugs)
    assert "is_trainable" in src and "is_backfilled" in src

def test_cache_key_includes_universe():
    src = inspect.getsource(ItemForecaster._voted_cache_key)
    assert "universe" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_universe_routing.py -q`
Expected: FAIL (`universe` not in signatures).

- [ ] **Step 3: Implement**

`_resolve_backfilled_slugs` (line 1787) — add `universe="serve"` and pick the column:

```python
    def _resolve_backfilled_slugs(self, universe: str = "serve") -> set:
        column = "is_trainable" if universe == "train" else "is_backfilled"
        try:
            slug_rows = self.db.execute(text(
                f"SELECT item_id FROM items WHERE {column} = 1")).fetchall()
            slugs = {r[0] for r in slug_rows}
            logger.info(f"  {universe.capitalize()} universe filter ({column}): {len(slugs)} items from DB")
            return slugs
        except Exception as e:
            logger.warning(f"  Could not fetch {column} items from DB, using all: {e}")
            import duckdb
            with duckdb.connect() as con:
                return {r[0] for r in con.sql(
                    "SELECT DISTINCT item_slug FROM read_parquet(?)",
                    params=[str(self.archive_dir / "prices-*.parquet")]).fetchall()}
```

`fetch_price_history` (line 1734) — add `universe="serve"`, pass to the resolve call (line 1739) and the cache key (line 1748):

```python
    def fetch_price_history(self, days_back: int = 365,
                            backfilled_only: bool = False,
                            universe: str = "serve") -> pd.DataFrame:
        ...
            backfilled_slugs = (self._resolve_backfilled_slugs(universe)
                                if backfilled_only else None)
            ...
            cache_key = self._voted_cache_key(days_back, backfilled_only, universe, ...)
```

`_voted_cache_key` (line 5049) — add `universe` to the signature and to the joined key list (line 5070), e.g. `f"universe={universe}"`.

`build_training_data` (line 4912) — add `universe="serve"` and pass it: `self.fetch_price_history(days_back=days_back, backfilled_only=backfilled_only, universe=universe)`.

`train()` (line 5315) — pass `universe="train"`: `self.build_training_data(days_back=1460, backfilled_only=True, universe="train", ...)`.

Leave `predict()` (line 7825) unchanged — it uses the default `universe="serve"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_universe_routing.py -q`
Expected: PASS. Also run the existing forecaster-touching tests to confirm no regression:
`venv/bin/python -m pytest tests/test_backtest_scoring.py tests/test_model_version_identity.py -q`

- [ ] **Step 5: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_universe_routing.py
git commit -m "feat: split train (is_trainable) and serve (is_backfilled) universes via universe param"
```

---

### Task 4: Training-size invariant guardrail test

**Files:**
- Test: `backend/tests/test_training_universe_guardrail.py`

**Interfaces:**
- Consumes: the `is_trainable` derivation (Task 2) and a synthetic fixture archive.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_training_universe_guardrail.py
import duckdb, pandas as pd

def test_trainable_excludes_iflow_only_items_at_scale(tmp_path):
    rows = []
    # 5 non-iflow items (trainable) + 5 iflow-only items (serve-only)
    for i in range(5):
        rows.append({"item_slug": f"Legit Item {i} (Factory New)", "day": "2024-01-01", "source": None, "mean_price": 10.0, "volume": None, "ingested_at": "2026-01-01"})
    for i in range(5):
        rows.append({"item_slug": f"Iflow Item {i} (Factory New)", "day": "2024-01-01", "source": "buff_iflow", "mean_price": 10.0, "volume": None, "ingested_at": "2026-01-01"})
    pd.DataFrame(rows).to_parquet(tmp_path / "prices-fixture-2024-01.parquet")
    con = duckdb.connect()
    trainable = {r[0] for r in con.sql(
        f"SELECT DISTINCT item_slug FROM read_parquet('{tmp_path}/prices-*.parquet', union_by_name=true) WHERE day < '2026-01-01' AND source IS DISTINCT FROM 'buff_iflow'").fetchall()}
    backfilled = {r[0] for r in con.sql(
        f"SELECT DISTINCT item_slug FROM read_parquet('{tmp_path}/prices-*.parquet', union_by_name=true) WHERE day < '2026-01-01'").fetchall()}
    con.close()
    assert len(trainable) == 5                 # training set does NOT grow with iflow
    assert len(backfilled) == 10               # serve set widens with iflow
    assert all("Iflow Item" not in s for s in trainable)
```

- [ ] **Step 2: Run test to verify it fails then passes**

Run: `venv/bin/python -m pytest tests/test_training_universe_guardrail.py -q`
Expected: PASS immediately (the predicate is validated). If it fails, the `source IS DISTINCT FROM` predicate in Task 2 is wrong — fix Task 2, not the test.

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_training_universe_guardrail.py
git commit -m "test: guardrail that is_trainable excludes iflow-only items"
```

---

### Task 5: Prod migration + iflow ingest + verification (OPS — not TDD)

**Files:**
- Data: `RayanR000/cs2-oracle-data` :: `price-archive/prices-buff_iflow-*.parquet` (50 files from `buff-iflow-staging/`)
- Prod DB: `items.is_trainable` column

This task has no unit test — it is an ops sequence with explicit verification gates. Do each step and confirm its check before the next.

- [ ] **Step 1: Confirm how prod sets `is_backfilled`.** Before touching prod, verify `init_local_db.py` (or its documented equivalent) is what seeds `items.is_backfilled` in prod. If prod uses a different path, mirror the `is_trainable` derivation there. Do not proceed until confirmed.

- [ ] **Step 2: Add the column in prod (read-then-write).** Against the prod Supabase `DATABASE_URL`, run:

```sql
ALTER TABLE items ADD COLUMN IF NOT EXISTS is_trainable INTEGER DEFAULT 0;
UPDATE items SET is_trainable = is_backfilled;   -- pre-ingest, train == current backfilled set
```

Verify: `SELECT COUNT(*) FROM items WHERE is_trainable = 1;` ≈ current backfilled count (~5,536), and the ≥$1 subset ≈ 926 (via the median-price gate). Training is unchanged at this point.

- [ ] **Step 3: Ingest iflow into the durable archive.** Add the 50 `prices-buff_iflow-*.parquet` files to `RayanR000/cs2-oracle-data` `price-archive/`. NOTE: CI writes that repo via orphan commit + force-push (`aggregator-update.yml`) — coordinate so the ingest is not clobbered by the next aggregator run (add on a branch and merge, or append within the aggregator's write). See memory [[panels-backed-up]] for the repo's write model.

- [ ] **Step 4: Re-derive flags in prod.** Run the derivation (Step 1's path) against the archive now containing iflow. Verify: `is_backfilled` count jumps (~+11k), `is_trainable` count stays ~5,536, and `is_trainable ∩ median≥$1` stays ≈ 926.

- [ ] **Step 5: Predict-only run + wall-time check.** Run the daily forecast in predict-only mode (`forecast_prices.py --predict-only`, no retrain). Verify: `item_forecasts` gains rows for the new items, `/opportunities` surfaces new ≥$1 items, and record the predict-step wall time against the 30-min daily cap (spec risk: wider feature build). If it breaches the cap, stop and revisit before enabling in the daily chain.

- [ ] **Step 6: Commit any code/doc changes** (the data-repo change commits in its own repo):

```bash
git add docs/superpowers/plans/2026-08-17-iflow-serve-universe-expansion.md
git commit -m "docs: iflow serve-universe expansion ingest runbook"
```

**Rollback:** remove the iflow files from the archive, re-derive flags — `is_backfilled` contracts to ~5,536 and the served set returns to today's. `is_trainable` is inert without iflow.

---

## Notes

- `icon_url` for new items is NOT on the forecast path; the trending surface (`items.py:131`) hides items without it, but `/opportunities` and per-item forecasts serve them regardless. Backfilling `icon_url` is an optional follow-up, out of scope here.
- The new items over-cover (95–97% vs 80% target) — same marginal-over-coverage problem the existing cohort has, extended. Do not claim calibrated 80% bands on them without the separate marginal-coverage fix.
