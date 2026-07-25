# Steam Price-History Gap Backfill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill the Apr 16 – Jul 8, 2026 gap in `price-archive/prices-2026.parquet` from the free `17mafo/cs-price-tracker` GitHub daily Steam snapshots, so the forecaster can be retrained on a complete serving window.

**Architecture:** One new script, `backend/scripts/merge_17mafo_gap.py`, modeled on the existing `merge_hf_dataset.py`. A pure, network-free transform (unit-tested with fixtures) converts each day's JSON into daily price rows; a thin fetch layer downloads + caches the 84 files; `main()` wires fetch → transform → idempotent append via the reused `db.parquet._append_parquet`. A final manual runbook task does the real fill + validation.

**Tech Stack:** Python 3, `requests` (fetch), `pandas` + `duckdb` (transform/append), `pytest` (tests). Target files are DuckDB-readable parquet under `price-archive/`.

## Global Constraints

- **Free only.** No paid API. Sole source is `https://raw.githubusercontent.com/17mafo/cs-price-tracker/main/static/prices/YYYY-MM-DD.json`.
- **Gap window:** `2026-04-16` … `2026-07-08` inclusive (84 days).
- **Source label (verbatim):** `aggregator_steam_17mafo`.
- **Dedup keys (verbatim):** `["item_slug", "day", "source"]`, `keep="last"`.
- **Price field:** use `steam.last_24h` as the day's price; skip `null`/missing. `volume = 0` (not provided; consistent with existing `aggregator_steam_*` rows).
- **No name→slug mapping:** JSON keys are already the archive's `item_slug` (Steam `market_hash_name`).
- **No currency conversion:** scale is seam-verified against `aggregator_steam_7d` (0.3% match).
- **Targets:** `price-archive/prices-2026.parquet` and `price-archive/snapshots-2026.parquet` (repo-root `price-archive/`, NOT `price-archive/ops/`).
- **Raw cache:** `price-archive/raw/17mafo/` — download once, re-runs read from cache.
- **Reuse, don't duplicate:** import `_append_parquet` from `db.parquet` (signature `(path: Path, new_data: pd.DataFrame, dedup_keys: list[str])`).
- **Column schemas (verbatim, match `merge_hf_dataset.py`):**
  - `PRICE_COLS = ["item_slug", "day", "source", "mean_price", "min_price", "max_price", "median_price", "volume"]`
  - `SNAP_COLS  = ["item_slug", "day", "source", "price", "volume"]`
- **Working directory:** run `python` / `pytest` commands from `backend/` (that's why verify scripts reference `../price-archive`). Run `git` commands from the repo root (that's why `git add` paths are repo-root-relative, e.g. `backend/scripts/...`, `price-archive/...`). Tests are `backend/tests/test_*.py`.

---

### Task 1: Pure transform (JSON → price + snapshot frames)

**Files:**
- Create: `backend/scripts/merge_17mafo_gap.py`
- Test: `backend/tests/test_merge_17mafo_gap.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `SOURCE = "aggregator_steam_17mafo"` (module constant)
  - `PRICE_COLS`, `SNAP_COLS` (module constants, values in Global Constraints)
  - `transform_day(day_obj: dict, day: str) -> pd.DataFrame` — returns a DataFrame with exactly `PRICE_COLS` columns; one row per item whose `steam.last_24h` is a non-null number; `mean_price == median_price == min_price == max_price == last_24h`; `volume == 0`; `source == SOURCE`; `day` is a `pandas.Timestamp` from the `YYYY-MM-DD` string.
  - `prices_to_snapshots(prices: pd.DataFrame) -> pd.DataFrame` — returns a DataFrame with exactly `SNAP_COLS` columns; `price` copied from `mean_price`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_merge_17mafo_gap.py
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import merge_17mafo_gap as m


SAMPLE_DAY = {
    "AK-47 | Redline (Field-Tested)": {
        "steam": {"last_24h": 47.63418, "last_7d": 47.37, "last_30d": 46.6,
                  "last_90d": 48.5, "last_ever": 45.99}
    },
    "★ Karambit | Doppler (Factory New)": {  # null recent price -> skipped
        "steam": {"last_24h": None, "last_7d": None, "last_30d": None,
                  "last_90d": None, "last_ever": 1000.0}
    },
    "Broken Item": {"steam": {}},            # no last_24h -> skipped
    "Weird Item": "not-a-dict",              # non-dict value -> skipped
}


def test_transform_day_columns_and_values():
    df = m.transform_day(SAMPLE_DAY, "2026-04-16")

    assert list(df.columns) == m.PRICE_COLS
    assert len(df) == 1  # only the valid Redline row survives

    row = df.iloc[0]
    assert row["item_slug"] == "AK-47 | Redline (Field-Tested)"
    assert row["day"] == pd.Timestamp("2026-04-16")
    assert row["source"] == "aggregator_steam_17mafo"
    assert row["mean_price"] == 47.63418
    assert row["median_price"] == 47.63418
    assert row["min_price"] == 47.63418
    assert row["max_price"] == 47.63418
    assert row["volume"] == 0


def test_prices_to_snapshots():
    prices = m.transform_day(SAMPLE_DAY, "2026-04-16")
    snaps = m.prices_to_snapshots(prices)

    assert list(snaps.columns) == m.SNAP_COLS
    assert len(snaps) == 1
    assert snaps.iloc[0]["price"] == 47.63418
    assert snaps.iloc[0]["volume"] == 0
    assert snaps.iloc[0]["source"] == "aggregator_steam_17mafo"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_merge_17mafo_gap.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'merge_17mafo_gap'` (file not created yet).

- [ ] **Step 3: Write minimal implementation**

```python
# backend/scripts/merge_17mafo_gap.py
#!/usr/bin/env python3
"""
Backfill the Apr 16 - Jul 8 2026 Steam price-history gap into the Parquet
archive from the free 17mafo/cs-price-tracker GitHub daily snapshots.

Each source file is static/prices/YYYY-MM-DD.json: a JSON object keyed by
Steam market_hash_name (== our item_slug) whose value is
{"steam": {"last_24h", "last_7d", "last_30d", "last_90d", "last_ever"}}.
We take last_24h as that day's Steam price.

Usage:
    python scripts/merge_17mafo_gap.py                 # full gap fill
    python scripts/merge_17mafo_gap.py --dry-run       # fetch + report, no write
    python scripts/merge_17mafo_gap.py --refresh       # re-download cached files
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

SOURCE = "aggregator_steam_17mafo"
PRICE_COLS = ["item_slug", "day", "source", "mean_price", "min_price",
              "max_price", "median_price", "volume"]
SNAP_COLS = ["item_slug", "day", "source", "price", "volume"]


def transform_day(day_obj: dict, day: str) -> pd.DataFrame:
    """Convert one day's 17mafo JSON object into PRICE_COLS rows."""
    ts = pd.Timestamp(day)
    rows = []
    for slug, val in day_obj.items():
        if not isinstance(val, dict):
            continue
        steam = val.get("steam")
        if not isinstance(steam, dict):
            continue
        price = steam.get("last_24h")
        if price is None:
            continue
        rows.append({
            "item_slug": slug,
            "day": ts,
            "source": SOURCE,
            "mean_price": price,
            "min_price": price,
            "max_price": price,
            "median_price": price,
            "volume": 0,
        })
    return pd.DataFrame(rows, columns=PRICE_COLS)


def prices_to_snapshots(prices: pd.DataFrame) -> pd.DataFrame:
    """Derive snapshot rows (one price per item/day) from price rows."""
    snaps = prices[["item_slug", "day", "source", "mean_price", "volume"]].copy()
    snaps = snaps.rename(columns={"mean_price": "price"})
    return snaps[SNAP_COLS]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_merge_17mafo_gap.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/merge_17mafo_gap.py backend/tests/test_merge_17mafo_gap.py
git commit -m "feat: pure transform for 17mafo Steam gap backfill"
```

---

### Task 2: Coverage validation

**Files:**
- Modify: `backend/scripts/merge_17mafo_gap.py`
- Test: `backend/tests/test_merge_17mafo_gap.py`

**Interfaces:**
- Consumes: `transform_day` (Task 1).
- Produces:
  - `gap_dates(start: str, end: str) -> list[str]` — inclusive list of `YYYY-MM-DD` strings.
  - `validate_coverage(prices: pd.DataFrame, expected_dates: list[str], min_items: int = 20000) -> None` — raises `AssertionError` if any expected date is missing from `prices["day"]`, or if any present date has fewer than `min_items` distinct `item_slug`. No return value on success.

- [ ] **Step 1: Write the failing test**

```python
# append to backend/tests/test_merge_17mafo_gap.py
import pytest


def _prices_for_dates(dates, n_items):
    frames = []
    for d in dates:
        obj = {f"Item {i}": {"steam": {"last_24h": 1.0 + i}} for i in range(n_items)}
        frames.append(m.transform_day(obj, d))
    return pd.concat(frames, ignore_index=True)


def test_gap_dates_inclusive():
    dates = m.gap_dates("2026-04-16", "2026-07-08")
    assert dates[0] == "2026-04-16"
    assert dates[-1] == "2026-07-08"
    assert len(dates) == 84


def test_validate_coverage_passes():
    dates = ["2026-04-16", "2026-04-17"]
    prices = _prices_for_dates(dates, n_items=25)
    m.validate_coverage(prices, dates, min_items=25)  # no raise


def test_validate_coverage_missing_day_raises():
    dates = ["2026-04-16", "2026-04-17"]
    prices = _prices_for_dates(["2026-04-16"], n_items=25)  # 04-17 missing
    with pytest.raises(AssertionError, match="2026-04-17"):
        m.validate_coverage(prices, dates, min_items=25)


def test_validate_coverage_low_count_raises():
    dates = ["2026-04-16"]
    prices = _prices_for_dates(dates, n_items=5)
    with pytest.raises(AssertionError, match="item count"):
        m.validate_coverage(prices, dates, min_items=25)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_merge_17mafo_gap.py -k "gap_dates or coverage" -v`
Expected: FAIL — `AttributeError: module 'merge_17mafo_gap' has no attribute 'gap_dates'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to backend/scripts/merge_17mafo_gap.py (below prices_to_snapshots)

def gap_dates(start: str, end: str) -> list[str]:
    """Inclusive list of YYYY-MM-DD date strings from start to end."""
    rng = pd.date_range(start=start, end=end, freq="D")
    return [d.strftime("%Y-%m-%d") for d in rng]


def validate_coverage(prices: pd.DataFrame, expected_dates: list[str],
                      min_items: int = 20000) -> None:
    """Raise AssertionError if any expected day is missing or too sparse."""
    present = {pd.Timestamp(d) for d in prices["day"].unique()}
    for d in expected_dates:
        ts = pd.Timestamp(d)
        assert ts in present, f"missing day {d} in backfill"
        count = prices.loc[prices["day"] == ts, "item_slug"].nunique()
        assert count >= min_items, (
            f"low item count for {d}: {count} < {min_items}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_merge_17mafo_gap.py -k "gap_dates or coverage" -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/merge_17mafo_gap.py backend/tests/test_merge_17mafo_gap.py
git commit -m "feat: gap-date enumeration and coverage validation for 17mafo backfill"
```

---

### Task 3: Fetch + cache one day

**Files:**
- Modify: `backend/scripts/merge_17mafo_gap.py`
- Test: `backend/tests/test_merge_17mafo_gap.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `RAW_URL_TEMPLATE = "https://raw.githubusercontent.com/17mafo/cs-price-tracker/main/static/prices/{date}.json"` (module constant)
  - `fetch_day(date: str, cache_dir: Path, refresh: bool = False, session=None) -> Path` — returns the local cached file path. If the file exists in `cache_dir` and `refresh` is False, does NOT hit the network. Otherwise GETs `RAW_URL_TEMPLATE`, raises `RuntimeError` on non-200, writes bytes to `cache_dir/<date>.json`. `session` defaults to a module-level `requests.Session`.
  - `load_day(path: Path) -> dict` — `json.load` of a cached file.

- [ ] **Step 1: Write the failing test**

```python
# append to backend/tests/test_merge_17mafo_gap.py
import json


class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.content = json.dumps(payload).encode()


class _FakeSession:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.payload = payload or {}
        self.calls = 0

    def get(self, url, timeout=None):
        self.calls += 1
        return _FakeResp(self.status_code, self.payload)


def test_fetch_day_downloads_when_absent(tmp_path):
    sess = _FakeSession(200, {"Item A": {"steam": {"last_24h": 5.0}}})
    path = m.fetch_day("2026-04-16", tmp_path, session=sess)
    assert path.exists()
    assert sess.calls == 1
    assert m.load_day(path)["Item A"]["steam"]["last_24h"] == 5.0


def test_fetch_day_uses_cache(tmp_path):
    (tmp_path / "2026-04-16.json").write_text('{"Item A": {"steam": {"last_24h": 9.0}}}')
    sess = _FakeSession(200, {})
    path = m.fetch_day("2026-04-16", tmp_path, session=sess)
    assert sess.calls == 0                      # cache hit, no network
    assert m.load_day(path)["Item A"]["steam"]["last_24h"] == 9.0


def test_fetch_day_raises_on_error(tmp_path):
    sess = _FakeSession(404, {})
    with pytest.raises(RuntimeError, match="404"):
        m.fetch_day("2026-04-16", tmp_path, session=sess)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_merge_17mafo_gap.py -k fetch -v`
Expected: FAIL — `AttributeError: module 'merge_17mafo_gap' has no attribute 'fetch_day'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to top imports of backend/scripts/merge_17mafo_gap.py
import json
import requests

# add near the other constants
RAW_URL_TEMPLATE = ("https://raw.githubusercontent.com/17mafo/cs-price-tracker/"
                    "main/static/prices/{date}.json")
_SESSION = requests.Session()


# add below validate_coverage
def load_day(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def fetch_day(date: str, cache_dir: Path, refresh: bool = False,
              session=None) -> Path:
    """Return local path to <date>.json, downloading + caching if needed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{date}.json"
    if path.exists() and not refresh:
        return path
    sess = session or _SESSION
    resp = sess.get(RAW_URL_TEMPLATE.format(date=date), timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"fetch {date} failed: HTTP {resp.status_code}")
    path.write_bytes(resp.content)
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_merge_17mafo_gap.py -k fetch -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/merge_17mafo_gap.py backend/tests/test_merge_17mafo_gap.py
git commit -m "feat: cached day fetcher for 17mafo backfill"
```

---

### Task 4: `main()` wiring — fetch all days, append, dry-run

**Files:**
- Modify: `backend/scripts/merge_17mafo_gap.py`
- Test: `backend/tests/test_merge_17mafo_gap.py`

**Interfaces:**
- Consumes: `transform_day`, `prices_to_snapshots`, `gap_dates`, `validate_coverage`, `fetch_day`, `load_day` (Tasks 1–3); `_append_parquet` from `db.parquet`.
- Produces:
  - `run(start: str, end: str, out_dir: Path, cache_dir: Path, dry_run: bool = False, refresh: bool = False, min_items: int = 20000, fetch=fetch_day) -> pd.DataFrame` — fetches every gap day, concatenates transformed prices, runs `validate_coverage`, and (unless `dry_run`) appends prices to `out_dir/prices-2026.parquet` and snapshots to `out_dir/snapshots-2026.parquet` via `_append_parquet` with the dedup keys. Returns the combined prices DataFrame. `fetch` is injectable for testing.
  - `main()` — argparse CLI (`--start-date`, `--end-date` default the gap bounds; `--out-dir` default `../price-archive`; `--dry-run`; `--refresh`) calling `run`.

- [ ] **Step 1: Write the failing test**

```python
# append to backend/tests/test_merge_17mafo_gap.py
import duckdb


def _fake_fetch_factory(payloads_by_date, cache_dir):
    """Return a fetch_day-compatible callable backed by in-memory payloads."""
    def _fetch(date, cdir, refresh=False, session=None):
        cdir.mkdir(parents=True, exist_ok=True)
        p = cdir / f"{date}.json"
        p.write_text(json.dumps(payloads_by_date[date]))
        return p
    return _fetch


def _payloads(dates, n_items):
    return {d: {f"Item {i}": {"steam": {"last_24h": float(i + 1)}}
                for i in range(n_items)} for d in dates}


def test_run_writes_parquet_and_is_idempotent(tmp_path):
    dates = ["2026-04-16", "2026-04-17"]
    payloads = _payloads(dates, n_items=30)
    out_dir = tmp_path / "price-archive"
    cache_dir = tmp_path / "raw"
    fetch = _fake_fetch_factory(payloads, cache_dir)

    m.run("2026-04-16", "2026-04-17", out_dir, cache_dir,
          min_items=30, fetch=fetch)

    prices_path = out_dir / "prices-2026.parquet"
    snaps_path = out_dir / "snapshots-2026.parquet"
    assert prices_path.exists() and snaps_path.exists()

    con = duckdb.connect()
    n1 = con.sql(f"SELECT COUNT(*) FROM read_parquet('{prices_path}')").fetchone()[0]
    assert n1 == 60  # 2 days x 30 items

    # re-run must not duplicate (dedup on item_slug, day, source)
    m.run("2026-04-16", "2026-04-17", out_dir, cache_dir,
          min_items=30, fetch=fetch)
    n2 = con.sql(f"SELECT COUNT(*) FROM read_parquet('{prices_path}')").fetchone()[0]
    assert n2 == 60


def test_run_dry_run_writes_nothing(tmp_path):
    dates = ["2026-04-16"]
    payloads = _payloads(dates, n_items=30)
    out_dir = tmp_path / "price-archive"
    cache_dir = tmp_path / "raw"
    fetch = _fake_fetch_factory(payloads, cache_dir)

    m.run("2026-04-16", "2026-04-16", out_dir, cache_dir,
          dry_run=True, min_items=30, fetch=fetch)
    assert not (out_dir / "prices-2026.parquet").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_merge_17mafo_gap.py -k run -v`
Expected: FAIL — `AttributeError: module 'merge_17mafo_gap' has no attribute 'run'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to imports of backend/scripts/merge_17mafo_gap.py
import argparse

from db.parquet import _append_parquet

DEFAULT_START = "2026-04-16"
DEFAULT_END = "2026-07-08"
DEDUP_KEYS = ["item_slug", "day", "source"]


def run(start: str, end: str, out_dir: Path, cache_dir: Path,
        dry_run: bool = False, refresh: bool = False,
        min_items: int = 20000, fetch=fetch_day) -> pd.DataFrame:
    dates = gap_dates(start, end)
    frames = []
    for d in dates:
        path = fetch(d, cache_dir, refresh=refresh)
        frames.append(transform_day(load_day(path), d))
    prices = pd.concat(frames, ignore_index=True)
    print(f"Transformed {len(prices):,} price rows over {len(dates)} days")

    validate_coverage(prices, dates, min_items=min_items)
    print("Coverage validation passed")

    if dry_run:
        print("Dry run — no files written")
        return prices

    out_dir.mkdir(parents=True, exist_ok=True)
    snapshots = prices_to_snapshots(prices)
    _append_parquet(out_dir / "prices-2026.parquet", prices[PRICE_COLS], DEDUP_KEYS)
    _append_parquet(out_dir / "snapshots-2026.parquet", snapshots[SNAP_COLS], DEDUP_KEYS)
    print(f"Done. Appended {start}..{end} as source={SOURCE}")
    return prices


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-date", default=DEFAULT_START)
    ap.add_argument("--end-date", default=DEFAULT_END)
    ap.add_argument("--out-dir", default="../price-archive")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "raw" / "17mafo"
    run(args.start_date, args.end_date, out_dir, cache_dir,
        dry_run=args.dry_run, refresh=args.refresh)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_merge_17mafo_gap.py -v`
Expected: PASS (all tests green).

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/merge_17mafo_gap.py backend/tests/test_merge_17mafo_gap.py
git commit -m "feat: main run wiring for 17mafo Steam gap backfill"
```

---

### Task 5: Live fill + validation gate (runbook)

This task is a manual runbook, not TDD — it runs the real backfill and verifies it against the live archive. Do NOT automate the seam check into pytest (it depends on live archive state).

**Files:**
- Modify (data): `price-archive/prices-2026.parquet`, `price-archive/snapshots-2026.parquet`
- Create (data): `price-archive/raw/17mafo/2026-04-16.json … 2026-07-08.json`

- [ ] **Step 1: Dry run against real source (downloads + validates, no write)**

Run (from `backend/`):
```bash
python scripts/merge_17mafo_gap.py --dry-run
```
Expected: prints `Transformed ~2,300,000 price rows over 84 days` (order of magnitude: ~27–28K items × 84 days) and `Coverage validation passed`. If validation raises, STOP and investigate the offending day before proceeding.

- [ ] **Step 2: Real fill**

Run:
```bash
python scripts/merge_17mafo_gap.py
```
Expected: `Done. Appended 2026-04-16..2026-07-08 as source=aggregator_steam_17mafo`.

- [ ] **Step 3: Verify day coverage in the archive**

Run:
```bash
python3 -c "
import duckdb
con=duckdb.connect()
p='../price-archive/prices-2026.parquet'
r=con.sql(f\"SELECT COUNT(DISTINCT day) d, MIN(day) mn, MAX(day) mx, COUNT(*) n FROM read_parquet('{p}') WHERE source='aggregator_steam_17mafo'\").df()
print(r.to_string())
gap=con.sql(f\"SELECT COUNT(DISTINCT CAST(day AS DATE)) FROM read_parquet('{p}') WHERE day BETWEEN '2026-04-16' AND '2026-07-08'\").fetchone()[0]
print('distinct days now present in gap window:', gap, '(expect 84)')
"
```
Expected: `d = 84`, `mn = 2026-04-16`, `mx = 2026-07-08`; distinct days in gap window `= 84`.

- [ ] **Step 4: Seam continuity spot-check**

Run:
```bash
python3 -c "
import duckdb
con=duckdb.connect()
p='../price-archive/prices-2026.parquet'
for item in ['AK-47 | Redline (Field-Tested)','AWP | Asiimov (Field-Tested)']:
    print('===', item, '===')
    q=f'''SELECT CAST(day AS DATE) d, source, median_price
          FROM read_parquet('{p}')
          WHERE item_slug='{item}'
            AND ((day BETWEEN '2026-04-14' AND '2026-04-18') OR (day BETWEEN '2026-07-07' AND '2026-07-11'))
          ORDER BY d, source'''
    print(con.sql(q).df().to_string())
"
```
Expected: no discontinuity across the Apr-15→16 and Jul-08→09 seams (the new `aggregator_steam_17mafo` prices sit within a plausible band of the adjacent real data; AK-Redline ~43–48 range).

- [ ] **Step 5: Commit cached raw files + note**

```bash
git add price-archive/raw/17mafo/
git commit -m "data: cache 17mafo Steam daily snapshots for Apr 16 - Jul 8 2026 gap fill"
```

Note: `prices-2026.parquet` / `snapshots-2026.parquet` are large data artifacts — follow the repo's existing convention for committing archive parquet (check `.gitignore` before adding them; if they are gitignored, the fill lives only on disk, which is the existing behavior).

---

## Downstream (out of scope for this plan)

After the fill validates, the separate retrain + `backtest_accuracy` task confirms the ~80% interval coverage predicted in memory `q90-goss-interval-bug-is-stale`. Not part of this plan.
