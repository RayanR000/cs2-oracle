# Recoverable-Data Sidecars Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover volume, BUFF bid, StatTrak premium, and retroactive supply-depth into a local research dataset as four `(item_id, date)` sidecar parquets, joined into feature engineering behind off-by-default flags, without touching the durable CI archive.

**Architecture:** Each feed is produced by a pure ingest/build script that writes one sidecar parquet under the local archive dir. A single new `_attach_sidecars` seam left-joins whatever sidecars exist onto the `daily` frame inside `engineer_features` (so both `train` and `predict` pick them up), keeping every recovered quantity out of price voting. Existing volume features are un-shelved; bid and StatTrak get new derived features. All gated so a missing sidecar or an off flag degrades to the current behaviour.

**Tech Stack:** Python 3.13 (local) / 3.11 (CI), pandas, DuckDB, LightGBM, pytest. Kaggle CLI (creds configured), `gh` API, `lzma` (stdlib) for the atalantus `.xz`.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-08-13-recoverable-data-sidecars-design.md`. Read it first.
- **`item_id` ≡ `item_slug` ≡ `market_hash_name`** — verified exact for all 4,342 recovered items; every join is a plain equijoin, no fuzzy matching.
- **Sidecars are local-only.** Write to `self.archive_dir` (the local `price-archive/`, gitignored). Never write to `cs2-oracle-data` or `aggregator-update.yml`. Never commit raw recovered rows (licenses: private/local training only).
- **Recovered quantities never vote as a price.** The join happens *after* voting, inside `engineer_features`; sidecars carry volume/bid/supply, never a consensus price.
- **Never zero-fill on bad input.** Parsers raise or drop malformed rows (the repo has shipped silent-zero bugs twice). A missing sidecar file is a no-op; a corrupt one raises.
- Run tests from `backend/` through the venv: `venv/bin/python -m pytest tests/<file> -q`.
- All new model-facing behaviour is behind an env flag, default off, mirroring `_lambdarank_enabled` (`forecaster.py:8359`): `return os.environ.get("FLAG") == "1"`.
- Sidecar schema is fixed across tasks — columns exactly: `item_id: str`, `date: datetime.date`, plus the feed column(s) below. Parquet, one row per `(item_id, date)`.

| Sidecar file | Columns beyond key | Built by |
|---|---|---|
| `volume-panel.parquet` | `steam_volume: int64`, `steam_sale_median: float64` | Task 1 |
| `supply-history.parquet` | `buff_listing_count: int64` | Task 2 |
| `bid-panel.parquet` | `buff_bid: float64` | Task 3 |
| `stattrak-panel.parquet` | `st_premium: float64` | Task 4 |

---

### Task 1: Volume panel ingest (devynpruden Kaggle parquet)

**Files:**
- Create: `backend/scripts/ingest_volume_panel.py`
- Test: `backend/tests/test_ingest_volume_panel.py`

**Interfaces:**
- Produces: `build_volume_panel(src_df: pd.DataFrame) -> pd.DataFrame` returning columns `["item_id","date","steam_volume","steam_sale_median"]`, one row per `(item_id, date)`; `SIDECAR_NAME = "volume-panel.parquet"`; `main(kaggle_parquet: Path, out_dir: Path) -> Path`.
- The source parquet schema (verified): `market_hash_name, date, price_median, volume, stattrak, exterior, weapon, ...`; source is `steam_ssr`; `volume` is dense `>0` 2013→2026-06-15.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_ingest_volume_panel.py
import sys
from pathlib import Path
from datetime import date
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.ingest_volume_panel import build_volume_panel  # noqa: E402


def _src():
    return pd.DataFrame({
        "market_hash_name": ["AK-47 | Redline (Field-Tested)", "AK-47 | Redline (Field-Tested)", "Sticker | X"],
        "date": [date(2026, 5, 1), date(2026, 5, 1), date(2026, 5, 1)],
        "price_median": [10.0, 12.0, 1.0],
        "volume": [5, 7, 0],
    })


def test_maps_name_to_item_id_and_keeps_volume():
    out = build_volume_panel(_src())
    assert list(out.columns) == ["item_id", "date", "steam_volume", "steam_sale_median"]
    row = out[out["item_id"] == "AK-47 | Redline (Field-Tested)"]
    # duplicate (item_id, date) collapses to one row
    assert len(row) == 1


def test_volume_is_integer_not_zero_filled():
    out = build_volume_panel(_src())
    assert out["steam_volume"].dtype.kind == "i"
    # the zero-volume sticker row is kept as 0, never dropped or nan-filled
    assert (out["steam_volume"] == 0).sum() == 1


def test_rejects_missing_required_column():
    bad = _src().drop(columns=["volume"])
    with pytest.raises((KeyError, ValueError)):
        build_volume_panel(bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_ingest_volume_panel.py -q`
Expected: FAIL with `ModuleNotFoundError`/`ImportError` (script not created).

- [ ] **Step 3: Write minimal implementation**

```python
# backend/scripts/ingest_volume_panel.py
"""Ingest the devynpruden CS2 daily-price parquet into a local volume sidecar.

Source: kaggle datasets download -d devynpruden/cs2-skin-price-history-2013-2026 \
        -f daily_prices.parquet
It reaches 2026-06-15 (covers the serving anchors) and carries dense `volume`.
Local research dataset only -- private/local training; do not redistribute rows.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

REQUIRED = ["market_hash_name", "date", "price_median", "volume"]
SIDECAR_NAME = "volume-panel.parquet"


def build_volume_panel(src_df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in src_df.columns]
    if missing:
        raise ValueError(f"volume source missing columns: {missing}")
    df = src_df[REQUIRED].copy()
    df = df.rename(columns={"market_hash_name": "item_id"})
    df = df.dropna(subset=["item_id", "date", "volume"])
    # One row per (item_id, date): mean median-price, summed volume.
    out = (df.groupby(["item_id", "date"], as_index=False)
             .agg(steam_volume=("volume", "sum"),
                  steam_sale_median=("price_median", "mean")))
    out["steam_volume"] = out["steam_volume"].astype("int64")
    return out[["item_id", "date", "steam_volume", "steam_sale_median"]]


def main(kaggle_parquet: Path, out_dir: Path) -> Path:
    src = pd.read_parquet(kaggle_parquet)
    panel = build_volume_panel(src)
    out = out_dir / SIDECAR_NAME
    panel.to_parquet(out, index=False)
    print(f"wrote {len(panel):,} rows, {panel['item_id'].nunique():,} items -> {out}")
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_ingest_volume_panel.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/ingest_volume_panel.py backend/tests/test_ingest_volume_panel.py
git commit -m "feat: volume-panel sidecar ingest from devynpruden parquet"
```

---

### Task 2: Supply-depth history ingest (atalantus xz-JSON)

**Files:**
- Create: `backend/scripts/ingest_supply_history.py`
- Test: `backend/tests/test_ingest_supply_history.py`

**Interfaces:**
- Produces: `parse_supply_history(raw: dict) -> pd.DataFrame` returning `["item_id","date","buff_listing_count"]`; `SIDECAR_NAME = "supply-history.parquet"`; `main(xz_path: Path, out_dir: Path) -> Path`.
- Source format (verified from repo README + memory): the decompressed `price-history-daily.json.xz` is `{name: [[unix_seconds...], [price_cny_x100...], [listing_count...]]}`. The **third** array is listing count, populated only after 2023-01-25 (earlier entries have no third array or nulls). Coverage ends 2024-01-19.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_ingest_supply_history.py
import sys
from pathlib import Path
from datetime import date
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.ingest_supply_history import parse_supply_history  # noqa: E402

# 2023-06-01 and 2023-06-02 in unix seconds
T1, T2 = 1685577600, 1685664000


def test_extracts_third_array_as_listing_count():
    raw = {"AK-47 | Redline (Field-Tested)": [[T1, T2], [1000, 1100], [42, 40]]}
    out = parse_supply_history(raw)
    assert list(out.columns) == ["item_id", "date", "buff_listing_count"]
    assert out.iloc[0]["buff_listing_count"] == 42
    assert out.iloc[0]["date"] == date(2023, 6, 1)


def test_drops_items_without_listing_counts_not_zero_fills():
    # An item present before 2023-01-25 has only two arrays (price, no count).
    raw = {"Old | Item": [[T1], [500]]}
    out = parse_supply_history(raw)
    assert len(out) == 0  # dropped, never zero-filled


def test_length_mismatch_raises():
    raw = {"Bad | Item": [[T1, T2], [1000, 1100], [42]]}  # counts shorter than timestamps
    with pytest.raises(ValueError):
        parse_supply_history(raw)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_ingest_supply_history.py -q`
Expected: FAIL (ImportError).

- [ ] **Step 3: Write minimal implementation**

```python
# backend/scripts/ingest_supply_history.py
"""Ingest atalantus/buff-price-history-archive listing counts into a sidecar.

Source: gh api repos/atalantus/buff-price-history-archive/contents/\
        price-history-daily.json.xz  (or clone). Decompress with lzma.
Format: {name: [[unix_s...],[cny*100...],[listing_count...]]}. Third array is
listing count, populated only after 2023-01-25; coverage ends 2024-01-19.
TRAINING-ONLY (no 2026 coverage). Private/local; do not redistribute rows.
"""
from __future__ import annotations
import json
import lzma
import sys
from datetime import date, timezone, datetime
from pathlib import Path
import pandas as pd

SIDECAR_NAME = "supply-history.parquet"


def parse_supply_history(raw: dict) -> pd.DataFrame:
    records = []
    for name, arrays in raw.items():
        if not isinstance(arrays, list) or len(arrays) < 3:
            continue  # no listing-count array -> drop, never zero-fill
        stamps, _prices, counts = arrays[0], arrays[1], arrays[2]
        if len(counts) != len(stamps):
            raise ValueError(f"{name}: counts/timestamps length mismatch")
        for ts, cnt in zip(stamps, counts):
            if cnt is None:
                continue
            d = datetime.fromtimestamp(ts, tz=timezone.utc).date()
            records.append((name, d, int(cnt)))
    out = pd.DataFrame(records, columns=["item_id", "date", "buff_listing_count"])
    if not out.empty:
        out = out.groupby(["item_id", "date"], as_index=False)["buff_listing_count"].max()
    return out


def main(xz_path: Path, out_dir: Path) -> Path:
    with lzma.open(xz_path, "rt", encoding="utf-8") as fh:
        raw = json.load(fh)
    panel = parse_supply_history(raw)
    out = out_dir / SIDECAR_NAME
    panel.to_parquet(out, index=False)
    print(f"wrote {len(panel):,} rows, {panel['item_id'].nunique():,} items -> {out}")
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_ingest_supply_history.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/ingest_supply_history.py backend/tests/test_ingest_supply_history.py
git commit -m "feat: supply-history sidecar ingest from atalantus archive"
```

---

### Task 3: Bid panel build (extract from the archive's own bid source)

**Files:**
- Create: `backend/scripts/build_bid_panel.py`
- Test: `backend/tests/test_build_bid_panel.py`

**Interfaces:**
- Produces: `build_bid_panel(bid_rows: pd.DataFrame) -> pd.DataFrame` returning `["item_id","date","buff_bid"]`; `SIDECAR_NAME = "bid-panel.parquet"`; `main(archive_dir: Path) -> Path`.
- Input rows come from the archive where `source == 'aggregator_buff163_buy'`; raw archive columns are `item_slug, day, source, mean_price, volume, ingested_at` (verified). The build renames `item_slug->item_id`, `day->date`, `mean_price->buff_bid`.
- ⚠️ **Two invariants bind on the raw-archive read** (both in `main`, not the pure function): (#1) read through `db/archive.py::prices_relation`, never a bare `read_parquet` glob — a bare glob takes the first file's schema and silently drops columns. (#2) apply `models/item_parser.py::archive_universe_sql_filter(source_column=None)` — this ANDs the phase-collapsed and phantom-slug filters (needed) while **omitting** the bid-source exclusion (which would otherwise drop the very rows we want; passing `source_column=None` skips it).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_build_bid_panel.py
import sys
from pathlib import Path
from datetime import date
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.build_bid_panel import build_bid_panel  # noqa: E402


def _rows():
    return pd.DataFrame({
        "item_slug": ["AK | X", "AK | X", "AK | X"],
        "day": [date(2026, 8, 1), date(2026, 8, 1), date(2026, 8, 2)],
        "mean_price": [7.0, 9.0, 8.0],
    })


def test_renames_keys_and_averages_within_day():
    out = build_bid_panel(_rows())
    assert list(out.columns) == ["item_id", "date", "buff_bid"]
    aug1 = out[(out["item_id"] == "AK | X") & (out["date"] == date(2026, 8, 1))]
    assert len(aug1) == 1
    assert aug1.iloc[0]["buff_bid"] == 8.0  # mean of 7 and 9


def test_drops_nonpositive_bids():
    rows = _rows()
    rows.loc[len(rows)] = ["AK | Y", date(2026, 8, 1), 0.0]
    out = build_bid_panel(rows)
    assert (out["item_id"] == "AK | Y").sum() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_build_bid_panel.py -q`
Expected: FAIL (ImportError).

- [ ] **Step 3: Write minimal implementation**

```python
# backend/scripts/build_bid_panel.py
"""Build a bid sidecar from the archive's own BUFF bid source.

`aggregator_buff163_buy` is BUFF's highest_order (a bid). The universe filter
excludes it from PRICE voting, so it is never a feature today -- this surfaces
it as a (item_id, date) panel for a bid/spread feature. Reads the local archive
directly via DuckDB; writes locally only.
"""
from __future__ import annotations
import sys
from pathlib import Path
import duckdb
import pandas as pd

from db.archive import prices_relation
from models.item_parser import archive_universe_sql_filter

BID_SOURCE = "aggregator_buff163_buy"
SIDECAR_NAME = "bid-panel.parquet"


def build_bid_panel(bid_rows: pd.DataFrame) -> pd.DataFrame:
    df = bid_rows.rename(columns={"item_slug": "item_id", "day": "date",
                                  "mean_price": "buff_bid"})
    df = df[df["buff_bid"] > 0]
    out = (df.groupby(["item_id", "date"], as_index=False)["buff_bid"].mean())
    return out[["item_id", "date", "buff_bid"]]


def main(archive_dir: Path) -> Path:
    con = duckdb.connect()
    # Invariant #1: typed reader, not a bare glob. Invariant #2: phase/phantom
    # slug filters via archive_universe_sql_filter, but source_column=None so the
    # bid-source EXCLUSION is skipped -- we want exactly the bid rows.
    rel = prices_relation(con, archive_dir=archive_dir,
                          columns=["item_slug", "day", "mean_price", "source"])
    uni = archive_universe_sql_filter(source_column=None)
    rows = con.sql(
        f"select item_slug, day, mean_price from {rel} "
        f"where source = '{BID_SOURCE}' and {uni}"
    ).df()
    panel = build_bid_panel(rows)
    out = archive_dir / SIDECAR_NAME
    panel.to_parquet(out, index=False)
    print(f"wrote {len(panel):,} rows, {panel['item_id'].nunique():,} items -> {out}")
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_build_bid_panel.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/build_bid_panel.py backend/tests/test_build_bid_panel.py
git commit -m "feat: bid-panel sidecar from archive BUFF bid source"
```

---

### Task 4: StatTrak premium panel (compute from paired prices)

**Files:**
- Create: `backend/scripts/build_stattrak_panel.py`
- Test: `backend/tests/test_build_stattrak_panel.py`

**Interfaces:**
- Produces: `build_stattrak_panel(prices: pd.DataFrame) -> pd.DataFrame` returning `["item_id","date","st_premium"]`, where `item_id` is the **StatTrak** slug and `st_premium = st_price / base_price` on the same day; `SIDECAR_NAME = "stattrak-panel.parquet"`; `base_name(slug: str) -> str | None` returning the non-ST twin slug for a `"StatTrak™ …"` slug, else None; `main(volume_panel: Path, out_dir: Path) -> Path`.
- Input `prices` columns: `["item_id","date","price"]`. Source of prices for the build is the `steam_sale_median` from the volume panel (single coherent source, full history) — pass it in as `price`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_build_stattrak_panel.py
import sys
from pathlib import Path
from datetime import date
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.build_stattrak_panel import base_name, build_stattrak_panel  # noqa: E402


def test_base_name_strips_stattrak_prefix():
    assert base_name("StatTrak™ AK-47 | Redline (Field-Tested)") == \
        "AK-47 | Redline (Field-Tested)"
    assert base_name("AK-47 | Redline (Field-Tested)") is None  # not ST


def test_premium_is_ratio_on_matched_day():
    prices = pd.DataFrame({
        "item_id": ["StatTrak™ AK-47 | Redline (Field-Tested)",
                    "AK-47 | Redline (Field-Tested)"],
        "date": [date(2026, 5, 1), date(2026, 5, 1)],
        "price": [21.0, 10.0],
    })
    out = build_stattrak_panel(prices)
    assert list(out.columns) == ["item_id", "date", "st_premium"]
    assert out.iloc[0]["item_id"].startswith("StatTrak")
    assert abs(out.iloc[0]["st_premium"] - 2.1) < 1e-9


def test_st_without_base_is_dropped():
    prices = pd.DataFrame({
        "item_id": ["StatTrak™ Only | Item"],
        "date": [date(2026, 5, 1)],
        "price": [50.0],
    })
    assert len(build_stattrak_panel(prices)) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_build_stattrak_panel.py -q`
Expected: FAIL (ImportError).

- [ ] **Step 3: Write minimal implementation**

```python
# backend/scripts/build_stattrak_panel.py
"""Compute a StatTrak-premium sidecar from paired ST / base prices.

No download: the ratio st_price/base_price on the same day is a free
usage/demand proxy. Prices come from the volume panel's steam_sale_median so
both legs share one coherent source. Private/local only.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ST_PREFIX = "StatTrak™ "
SIDECAR_NAME = "stattrak-panel.parquet"


def base_name(slug: str) -> str | None:
    return slug[len(ST_PREFIX):] if slug.startswith(ST_PREFIX) else None


def build_stattrak_panel(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices[prices["price"] > 0].copy()
    st = df[df["item_id"].str.startswith(ST_PREFIX)].copy()
    st["base_id"] = st["item_id"].map(base_name)
    merged = st.merge(
        df.rename(columns={"item_id": "base_id", "price": "base_price"}),
        on=["base_id", "date"], how="inner")
    merged["st_premium"] = merged["price"] / merged["base_price"]
    return merged[["item_id", "date", "st_premium"]].reset_index(drop=True)


def main(volume_panel: Path, out_dir: Path) -> Path:
    vp = pd.read_parquet(volume_panel)
    prices = vp.rename(columns={"steam_sale_median": "price"})[["item_id", "date", "price"]]
    panel = build_stattrak_panel(prices)
    out = out_dir / SIDECAR_NAME
    panel.to_parquet(out, index=False)
    print(f"wrote {len(panel):,} rows, {panel['item_id'].nunique():,} items -> {out}")
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_build_stattrak_panel.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/build_stattrak_panel.py backend/tests/test_build_stattrak_panel.py
git commit -m "feat: stattrak-premium sidecar from paired prices"
```

---

### Task 5: Sidecar join seam in `engineer_features`

**Files:**
- Modify: `backend/models/forecaster.py` — add `_attach_sidecars`, call it in `engineer_features` right after the `daily` frame is built (~line 3991), bump `ENGINEERED_CACHE_VERSION` 2→3 (~line 733).
- Test: `backend/tests/test_sidecar_attach.py`

**Interfaces:**
- Consumes: sidecar parquets in `self.archive_dir` from Tasks 1-4, each keyed `(item_id, date)`.
- Produces: `ItemForecaster._attach_sidecars(self, daily: pd.DataFrame) -> pd.DataFrame` — left-joins every sidecar present, overwrites `daily["volume"]` with `steam_volume` where present (else keeps existing), adds `buff_bid`/`st_premium`/`buff_listing_count` columns (NaN where absent). Missing file → that column is skipped entirely. Returns the widened frame.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_sidecar_attach.py
import sys
from pathlib import Path
from datetime import date
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.forecaster import ItemForecaster  # noqa: E402


def _daily():
    return pd.DataFrame({
        "item_id": ["AK | X", "AK | X"],
        "date": [date(2026, 5, 1), date(2026, 5, 2)],
        "price": [10.0, 11.0],
        "volume": [0, 0],  # archive volume is dead
    })


def test_volume_overwritten_from_sidecar(tmp_path, monkeypatch):
    vp = pd.DataFrame({"item_id": ["AK | X"], "date": [date(2026, 5, 1)],
                       "steam_volume": [55], "steam_sale_median": [10.0]})
    vp.to_parquet(tmp_path / "volume-panel.parquet", index=False)
    f = ItemForecaster.__new__(ItemForecaster)
    f.archive_dir = tmp_path
    out = f._attach_sidecars(_daily())
    assert out.loc[out["date"] == date(2026, 5, 1), "volume"].iloc[0] == 55
    assert out.loc[out["date"] == date(2026, 5, 2), "volume"].iloc[0] == 0  # unmatched keeps 0


def test_missing_sidecar_is_noop(tmp_path):
    f = ItemForecaster.__new__(ItemForecaster)
    f.archive_dir = tmp_path  # empty dir
    out = f._attach_sidecars(_daily())
    assert list(out.columns) == ["item_id", "date", "price", "volume"]
    assert "buff_bid" not in out.columns


def test_bid_column_added_when_present(tmp_path):
    bp = pd.DataFrame({"item_id": ["AK | X"], "date": [date(2026, 5, 1)], "buff_bid": [8.5]})
    bp.to_parquet(tmp_path / "bid-panel.parquet", index=False)
    f = ItemForecaster.__new__(ItemForecaster)
    f.archive_dir = tmp_path
    out = f._attach_sidecars(_daily())
    assert out.loc[out["date"] == date(2026, 5, 1), "buff_bid"].iloc[0] == 8.5
    assert pd.isna(out.loc[out["date"] == date(2026, 5, 2), "buff_bid"].iloc[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_sidecar_attach.py -q`
Expected: FAIL with `AttributeError: _attach_sidecars`.

- [ ] **Step 3: Write minimal implementation**

Add the method to `ItemForecaster` (place near `engineer_features`):

```python
    # Recovered demand/supply sidecars, joined AFTER voting so none of them
    # votes as a price. Local research dataset only; a missing file is a no-op.
    _SIDECARS = {
        "volume-panel.parquet": ["steam_volume", "steam_sale_median"],
        "bid-panel.parquet": ["buff_bid"],
        "stattrak-panel.parquet": ["st_premium"],
        "supply-history.parquet": ["buff_listing_count"],
    }

    def _attach_sidecars(self, daily: pd.DataFrame) -> pd.DataFrame:
        for fname, cols in self._SIDECARS.items():
            path = self.archive_dir / fname
            if not path.exists():
                continue
            side = pd.read_parquet(path)[["item_id", "date"] + cols]
            daily = daily.merge(side, on=["item_id", "date"], how="left")
            if "steam_volume" in cols:
                # Prefer recovered volume; keep existing where unmatched.
                daily["volume"] = daily["steam_volume"].fillna(daily["volume"])
                daily = daily.drop(columns=["steam_volume"])
        return daily
```

Call it in `engineer_features` immediately after the `daily = ...` block (~line 3999, before `_compute_price_features`):

```python
        else:
            daily = price_df
        daily = self._attach_sidecars(daily)   # <-- add this line
```

Bump the cache constant (~line 733):

```python
    ENGINEERED_CACHE_VERSION = 3   # v3: sidecar columns join into the daily frame
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_sidecar_attach.py tests/test_forecaster.py -q`
Expected: PASS (new file + existing forecaster tests unbroken).

- [ ] **Step 5: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_sidecar_attach.py
git commit -m "feat: join recovered sidecars into engineer_features (cache v3)"
```

---

### Task 6: Un-shelve volume features behind `VOLUME_FEATURES`

**Files:**
- Modify: `backend/models/forecaster.py` — add `_volume_features_enabled` (near `_lambdarank_enabled`, ~8359), add `VOLUME_FEATURE_NAMES`, add `_active_shelved_features`, and use it at the `_select_feature_cols` call (~line 4659) instead of `self.SHELVED_FEATURES`.
- Test: `backend/tests/test_volume_feature_flag.py`

**Interfaces:**
- Consumes: `daily["volume"]` (real, from Task 5).
- Produces: `ItemForecaster._volume_features_enabled() -> bool`; `ItemForecaster._active_shelved_features(self) -> frozenset[str]` — `SHELVED_FEATURES` minus the volume feature names when the flag is on.
- The 11 volume feature names are the columns `_compute_volume_features` adds; capture them as `VOLUME_FEATURE_NAMES` (read them off `_compute_volume_features` output on a dummy frame if unsure).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_volume_feature_flag.py
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.forecaster import ItemForecaster  # noqa: E402


def test_volume_names_shelved_by_default(monkeypatch):
    monkeypatch.delenv("VOLUME_FEATURES", raising=False)
    f = ItemForecaster.__new__(ItemForecaster)
    active = f._active_shelved_features()
    assert ItemForecaster.VOLUME_FEATURE_NAMES <= active  # still shelved


def test_flag_unshelves_volume_names(monkeypatch):
    monkeypatch.setenv("VOLUME_FEATURES", "1")
    f = ItemForecaster.__new__(ItemForecaster)
    active = f._active_shelved_features()
    assert not (ItemForecaster.VOLUME_FEATURE_NAMES & active)  # none still shelved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_volume_feature_flag.py -q`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Write minimal implementation**

```python
    VOLUME_FEATURE_NAMES = frozenset({
        "volume_7d", "volume_30d", "volume_ratio_7_30", "volume_change_7d",
        "volume_zscore_30d", "volume_trend_14d", "volume_spike",
        "log_volume", "volume_ma_7d", "volume_ma_30d", "volume_missing",
    })  # verify against _compute_volume_features output; adjust to the exact set

    @staticmethod
    def _volume_features_enabled() -> bool:
        return os.environ.get("VOLUME_FEATURES") == "1"

    def _active_shelved_features(self) -> frozenset:
        if self._volume_features_enabled():
            return self.SHELVED_FEATURES - self.VOLUME_FEATURE_NAMES
        return self.SHELVED_FEATURES
```

At the `_select_feature_cols` call (~line 4659), replace `self.SHELVED_FEATURES` with `self._active_shelved_features()`.

⚠️ **Verify `VOLUME_FEATURE_NAMES` against the real output** before committing: run `_compute_volume_features` on a dummy grouped frame and diff its added columns against `SHELVED_FEATURES` — the set must match exactly, or an un-shelve leaves stragglers.

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_volume_feature_flag.py tests/test_forecaster.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_volume_feature_flag.py
git commit -m "feat: VOLUME_FEATURES flag un-shelves the recovered volume features"
```

---

### Task 7: Bid features behind `BID_FEATURES`

**Files:**
- Modify: `backend/models/forecaster.py` — add `_bid_features_enabled`, add `_compute_bid_features(df)`, call it in `engineer_features` after `_compute_volume_features` when the flag is on.
- Test: `backend/tests/test_bid_features.py`

**Interfaces:**
- Consumes: `df["buff_bid"]` (from Task 5, NaN where absent), `df["price"]` (served ask).
- Produces: `ItemForecaster._bid_features_enabled() -> bool`; `_compute_bid_features(df) -> pd.DataFrame` adding `bid_ask_spread` (`(price - buff_bid)/price`) and `bid_present` (`buff_bid.notna().astype(int)`). Safe when `buff_bid` is entirely absent (returns df unchanged with `bid_present=0`).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_bid_features.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.forecaster import ItemForecaster  # noqa: E402


def test_spread_computed_when_bid_present():
    df = pd.DataFrame({"price": [10.0, 20.0], "buff_bid": [8.0, np.nan]})
    out = ItemForecaster._compute_bid_features(df.copy())
    assert abs(out.iloc[0]["bid_ask_spread"] - 0.2) < 1e-9
    assert out.iloc[0]["bid_present"] == 1
    assert out.iloc[1]["bid_present"] == 0


def test_no_bid_column_is_safe():
    df = pd.DataFrame({"price": [10.0]})
    out = ItemForecaster._compute_bid_features(df.copy())
    assert out.iloc[0]["bid_present"] == 0
    assert pd.isna(out.iloc[0]["bid_ask_spread"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_bid_features.py -q`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Write minimal implementation**

```python
    @staticmethod
    def _bid_features_enabled() -> bool:
        return os.environ.get("BID_FEATURES") == "1"

    @staticmethod
    def _compute_bid_features(df: pd.DataFrame) -> pd.DataFrame:
        if "buff_bid" not in df.columns:
            df["buff_bid"] = np.nan
        df["bid_present"] = df["buff_bid"].notna().astype(int)
        df["bid_ask_spread"] = (df["price"] - df["buff_bid"]) / df["price"]
        return df
```

In `engineer_features`, after the `_compute_volume_features` call:

```python
        df = self._compute_volume_features(df, grouped)
        if self._bid_features_enabled():
            df = self._compute_bid_features(df)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_bid_features.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_bid_features.py
git commit -m "feat: BID_FEATURES flag adds bid/ask-spread features"
```

---

### Task 8: StatTrak-premium feature behind `STATTRAK_FEATURE`

**Files:**
- Modify: `backend/models/forecaster.py` — add `_stattrak_feature_enabled`, add `_compute_stattrak_feature(df)`, call it in `engineer_features` alongside the bid features.
- Test: `backend/tests/test_stattrak_feature.py`

**Interfaces:**
- Consumes: `df["st_premium"]` (from Task 5, NaN where absent).
- Produces: `ItemForecaster._stattrak_feature_enabled() -> bool`; `_compute_stattrak_feature(df) -> pd.DataFrame` adding `st_premium` (kept, NaN→0 not applied — the model reads NaN as missing) and `st_premium_present` indicator. Safe when the column is absent.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_stattrak_feature.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.forecaster import ItemForecaster  # noqa: E402


def test_present_indicator():
    df = pd.DataFrame({"st_premium": [2.1, np.nan]})
    out = ItemForecaster._compute_stattrak_feature(df.copy())
    assert out.iloc[0]["st_premium_present"] == 1
    assert out.iloc[1]["st_premium_present"] == 0


def test_absent_column_is_safe():
    df = pd.DataFrame({"price": [10.0]})
    out = ItemForecaster._compute_stattrak_feature(df.copy())
    assert out.iloc[0]["st_premium_present"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_stattrak_feature.py -q`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
    @staticmethod
    def _stattrak_feature_enabled() -> bool:
        return os.environ.get("STATTRAK_FEATURE") == "1"

    @staticmethod
    def _compute_stattrak_feature(df: pd.DataFrame) -> pd.DataFrame:
        if "st_premium" not in df.columns:
            df["st_premium"] = np.nan
        df["st_premium_present"] = df["st_premium"].notna().astype(int)
        return df
```

In `engineer_features`, beside the bid call:

```python
        if self._stattrak_feature_enabled():
            df = self._compute_stattrak_feature(df)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_stattrak_feature.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_stattrak_feature.py
git commit -m "feat: STATTRAK_FEATURE flag adds usage-premium feature"
```

---

### Task 9: Build all sidecars locally + coverage verification

**Files:**
- Create: `backend/scripts/build_all_sidecars.py` (thin orchestrator over Tasks 1-4 mains)
- Test: `backend/tests/test_sidecar_coverage.py` (guards the schema/coverage contract, not the downloads)

**Interfaces:**
- Consumes: the four `main()` entrypoints; the downloaded source files (Kaggle parquet already in `$CLAUDE_JOB_DIR/tmp/daily_prices.parquet`; atalantus xz fetched via `gh api`).
- Produces: `run(archive_dir, volume_src, supply_src) -> dict[str,int]` mapping sidecar name → row count.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_sidecar_coverage.py
import sys
from pathlib import Path
from datetime import date
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.build_bid_panel import build_bid_panel
from scripts.build_stattrak_panel import build_stattrak_panel


def test_every_sidecar_has_the_join_key():
    # Contract: every sidecar exposes exactly (item_id, date) + its feed cols.
    bid = build_bid_panel(pd.DataFrame({
        "item_slug": ["AK | X"], "day": [date(2026, 8, 1)], "mean_price": [8.0]}))
    assert {"item_id", "date"} <= set(bid.columns)
    st = build_stattrak_panel(pd.DataFrame({
        "item_id": ["StatTrak™ AK | X", "AK | X"],
        "date": [date(2026, 8, 1), date(2026, 8, 1)], "price": [16.0, 8.0]}))
    assert {"item_id", "date"} <= set(st.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_sidecar_coverage.py -q`
Expected: FAIL only if imports break; otherwise write `build_all_sidecars.py` next and keep this green.

- [ ] **Step 3: Write the orchestrator**

```python
# backend/scripts/build_all_sidecars.py
"""Build all four local sidecars. Downloads are the caller's job (see docstrings):
  kaggle datasets download -d devynpruden/cs2-skin-price-history-2013-2026 -f daily_prices.parquet
  gh api repos/atalantus/buff-price-history-archive/contents/price-history-daily.json.xz
"""
from __future__ import annotations
import sys
from pathlib import Path
from scripts.ingest_volume_panel import main as volume_main, SIDECAR_NAME as VOL
from scripts.ingest_supply_history import main as supply_main
from scripts.build_bid_panel import main as bid_main
from scripts.build_stattrak_panel import main as st_main
import pandas as pd


def run(archive_dir: Path, volume_src: Path, supply_src: Path) -> dict:
    volume_main(volume_src, archive_dir)
    supply_main(supply_src, archive_dir)
    bid_main(archive_dir)
    st_main(archive_dir / VOL, archive_dir)
    return {p.name: len(pd.read_parquet(p))
            for p in archive_dir.glob("*-panel.parquet")}


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
```

- [ ] **Step 4: Run everything + build the real sidecars, then verify cohort coverage**

Run the suite:
`venv/bin/python -m pytest tests/test_ingest_volume_panel.py tests/test_ingest_supply_history.py tests/test_build_bid_panel.py tests/test_build_stattrak_panel.py tests/test_sidecar_attach.py tests/test_volume_feature_flag.py tests/test_bid_features.py tests/test_stattrak_feature.py tests/test_sidecar_coverage.py -q`

Then build for real and check how much of the 926-item ≥$1 cohort each sidecar covers (DuckDB one-liner against `price-archive/`), and confirm `steam_volume` is non-null on the tied serving anchors (2026-04→06). Record the coverage numbers — they bound every downstream A/B.

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/build_all_sidecars.py backend/tests/test_sidecar_coverage.py
git commit -m "feat: sidecar build orchestrator + coverage contract test"
```

---

## Measurement (runbook, not code — the gate)

The features ship gated off. Measuring them uses the **existing** paired-A/B harnesses, not new code, and is the promotion gate the spec requires (tied ≥$1 cohort, MDE computed first):

1. **Volume** — `scripts/ab_test_volume_features.py` with `VOLUME_FEATURES=1`; it is the priority because it is the only recovered feed that reaches the serving anchors. ⚠️ Confirm the harness admits the volume feature group past `_apply_feature_allowlist` (it measures a wider model than production's 33 columns).
2. **Bid / StatTrak** — a paired arm with `BID_FEATURES=1` / `STATTRAK_FEATURE=1`; read the tied-cohort edge, placebo before treatment.
3. **Supply-depth history** — training-only; a CV-rank-IC diagnostic on the 2023–2024 folds. Do not read a serving claim from it.

No feed is proposed for `cs2-oracle-data` until it clears its bar.

## Self-review notes

- **Spec coverage:** volume (T1,5,6), supply-depth history (T2,5), bid (T3,5,7), StatTrak (T4,5,8), sidecar/local-only/out-of-voting (T5), off-by-default flags (T6-8), measurement gate (runbook). All spec sections map to a task.
- **Availability-leakage** (spec caveat) is enforced by T9 step 4 (confirm non-null on the tied serving anchors before any serving claim) and by the flags defaulting off.
- **VOLUME_FEATURE_NAMES is the one place needing verification against live output** — flagged inline in Task 6 with the exact check to run.
