# Historical Price-Source Import Implementation Plan

> # ✅ ALL 7 TASKS LANDED (2026-08-09) — ⛔ promotion into the real archive is BLOCKED
>
> Evidence: `backend/collectors/price_history_import.py` (`a657db7`),
> `collectors/price_history_sources/cs2_prices_tracker.py` (`52f4b3b`,
> `STEAM_FEE_MULTIPLIER = 1.1607`), `scripts/import_price_history_source.py` (`99636e9`). The real
> staged import ran: **5,153 items / 1,963,626 rows**, full suite 1,657 pass. Ledger:
> `.superpowers/sdd/2026-08-08-price-history-source-import/progress.md`.
>
> **Outcome: `docs/changelog/2026-08-09-price-history-import-staged.md`.**
>
> ⛔ **Do not promote the staged archive.** Four review blockers are open and no commit after
> `52f4b3b` addresses them: the end seam is unmeasured; the gate's consensus is not production's
> consensus; `_preserve_first_arrival` NULLs `ingested_at` on non-`RangeIndex` frames; and the
> importer can under-import silently. Building is done; shipping is not.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import `LukeX404/cs2-prices-tracker`'s daily Steam price history (2025-02-17 → 2026-03-31) into a **staging** price archive as source `tracker_steam_24h`, so items that entered the archive in 2026-03 gain the pre-2026 rows the `is_backfilled` gate tests for.

**Architecture:** A source-specific adapter supplies only *how to fetch a day* and *how to parse it*. Everything downstream — stall detection, the gap quality gate, schema normalisation, the write — is shared, because a second backfill source is expected. Writes go through the existing `db/parquet.py::append_monthly`, exactly as `scripts/merge_hf_dataset.py` does.

**Tech Stack:** Python 3.13 (3.11 in CI), pandas, DuckDB, requests, pytest. Run everything from `backend/` through `venv/bin/python`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-08-price-history-source-import-design.md`. Every value below is copied from it verbatim.
- Source label is exactly `tracker_steam_24h`. It is **not** added to `BID_SOURCES` — it votes.
- Date range is **2025-02-17 → 2026-03-31** inclusive.
- Gap gate is **max interior gap ≤ 7 days** (`MAX_WINDOW_SPAN_DAYS`), measured between consecutive observations inside the imported range. Leading/trailing edges are not penalised.
- `volume` is **NULL, never 0**. A real zero never occurs in this archive.
- `ingested_at` is the **run's wall clock**, not `day`.
- No price floor at import. Price cohort selection belongs to `TRAIN_MIN_MEDIAN_PRICE`.
- Output goes to a **staging** dir (`--out-dir`, default `../archive-staging`). Never `price-archive/`, never `cs2-oracle-data`.
- Canonical columns and types come from `db/archive.py`: `item_slug VARCHAR, day DATE, source VARCHAR, mean_price DOUBLE, volume BIGINT, ingested_at TIMESTAMP`.
- Run tests as `venv/bin/python -m pytest tests/test_price_history_import.py -q` from `backend/`. Never a bare `pytest -q` — it collects `scripts/test_social_signal.py` and aborts on a missing `thefuzz`.
- The upstream repo has no LICENSE. Import code may be committed to the public repo; imported **data** may not.

---

## File Structure

| File | Responsibility |
|---|---|
| `backend/collectors/price_history_sources/__init__.py` | Adapter registry: name → module. Nothing else. |
| `backend/collectors/price_history_sources/cs2_prices_tracker.py` | THIS source only: `SOURCE`, `day_url()`, `parse_day()`. |
| `backend/collectors/price_history_import.py` | Generic: stall detection, gap gate, archive-frame construction. |
| `backend/scripts/import_price_history_source.py` | CLI: fetch loop, cache, orchestration, report. |
| `backend/tests/test_price_history_import.py` | All tests for the above. |

---

### Task 1: The cs2-prices-tracker adapter

**Files:**
- Create: `backend/collectors/price_history_sources/__init__.py`
- Create: `backend/collectors/price_history_sources/cs2_prices_tracker.py`
- Test: `backend/tests/test_price_history_import.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `SOURCE: str`, `day_url(day: date) -> str`, `parse_day(payload: dict, day: date) -> list[tuple[str, date, float]]`. Later tasks consume `parse_day`'s return type as the canonical "record" triple `(item_slug, day, price)`.

- [x] **Step 1: Write the failing tests**

Create `backend/tests/test_price_history_import.py`:

```python
"""Tests for the historical price-source import."""
from datetime import date, datetime

import pytest

from collectors.price_history_sources import cs2_prices_tracker as tracker


def test_source_label_is_exact():
    assert tracker.SOURCE == "tracker_steam_24h"


def test_day_url_uses_the_dated_price_file():
    url = tracker.day_url(date(2025, 6, 15))
    assert url == (
        "https://raw.githubusercontent.com/LukeX404/cs2-prices-tracker/"
        "main/static/prices/date/2025-06-15.json"
    )


def test_parse_day_reads_last_24h_only():
    payload = {
        "AK-47 | Redline (Field-Tested)": {
            "steam": {"last_24h": 12.5, "last_7d": 99.0, "last_30d": 98.0}
        }
    }
    assert tracker.parse_day(payload, date(2025, 6, 15)) == [
        ("AK-47 | Redline (Field-Tested)", date(2025, 6, 15), 12.5)
    ]


def test_parse_day_treats_null_last_24h_as_an_absent_row():
    payload = {
        "Sticker | Sherry": {"steam": {"last_24h": None, "last_7d": 4.0}},
        "AK-47 | Redline (Field-Tested)": {"steam": {"last_24h": 12.5}},
    }
    records = tracker.parse_day(payload, date(2025, 6, 15))
    assert [r[0] for r in records] == ["AK-47 | Redline (Field-Tested)"]


def test_parse_day_drops_non_positive_prices_rather_than_writing_zero():
    payload = {
        "Zero Item": {"steam": {"last_24h": 0.0}},
        "Negative Item": {"steam": {"last_24h": -1.0}},
        "AK-47 | Redline (Field-Tested)": {"steam": {"last_24h": 12.5}},
    }
    records = tracker.parse_day(payload, date(2025, 6, 15))
    assert [r[0] for r in records] == ["AK-47 | Redline (Field-Tested)"]


def test_parse_day_tolerates_a_missing_or_null_steam_object():
    payload = {
        "No Steam Key": {"buff": {"price": 3.0}},
        "Null Steam": {"steam": None},
        "Null Item": None,
        "AK-47 | Redline (Field-Tested)": {"steam": {"last_24h": 12.5}},
    }
    records = tracker.parse_day(payload, date(2025, 6, 15))
    assert [r[0] for r in records] == ["AK-47 | Redline (Field-Tested)"]
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_price_history_import.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'collectors.price_history_sources'`

- [x] **Step 3: Write the registry**

Create `backend/collectors/price_history_sources/__init__.py`:

```python
"""Adapters for one-shot historical price-source imports.

An adapter supplies two things and nothing else: how to fetch one day, and how
to parse that day into ``(item_slug, day, price)`` records. Stall detection, the
gap quality gate, schema normalisation and the write are shared across sources
and live in ``collectors/price_history_import.py``.

A second backfill source is expected; add a module here and register it below.
"""
from . import cs2_prices_tracker

ADAPTERS = {
    "cs2_prices_tracker": cs2_prices_tracker,
}
```

- [x] **Step 4: Write the adapter**

Create `backend/collectors/price_history_sources/cs2_prices_tracker.py`:

```python
"""`LukeX404/cs2-prices-tracker` — dated daily Steam price files on GitHub.

The repo commits one JSON per day under ``static/prices/date/``. Schema is
``{market_hash_name: {steam: {last_24h, last_7d, last_30d, last_90d}}}``.

**Only `last_24h` is read.** It is Steam's 24-hour average sale price — the same
quantity as ``aggregator_sync``'s primary field, so it needs no basis conversion
and is a legitimate ask-side vote. ``last_7d``/``30d``/``90d`` are trailing-window
means and are exactly the feeds documented as voting against point-in-time asks;
importing them would repeat that error.

Keys are the raw ``market_hash_name``, which is what the archive uses as
``item_slug`` (verified 2026-08-08: 0 key-format failures across 24,238 names).
"""
from datetime import date

#: Distinct label so the rows stay attributable and a read-time filter can drop
#: them without rewriting the archive. Not a bid, so NOT in ``BID_SOURCES``.
SOURCE = "tracker_steam_24h"

_BASE = (
    "https://raw.githubusercontent.com/LukeX404/cs2-prices-tracker/"
    "main/static/prices/date"
)


def day_url(day: date) -> str:
    """The raw-content URL for one dated price file."""
    return f"{_BASE}/{day.isoformat()}.json"


def parse_day(payload: dict, day: date) -> list[tuple[str, date, float]]:
    """``(item_slug, day, price)`` for every item priced on *day*.

    A null ``last_24h`` is an ABSENT row, not a zero — the item simply did not
    trade. Non-positive prices are dropped for the same reason: a fabricated
    zero is what defeated ``has_volume``/``volume_missing`` and shelved eleven
    features, and the same trap applies to price.
    """
    records: list[tuple[str, date, float]] = []
    for name, entry in (payload or {}).items():
        steam = (entry or {}).get("steam") or {}
        price = steam.get("last_24h")
        if price is None:
            continue
        price = float(price)
        if price <= 0:
            continue
        records.append((name, day, price))
    return records
```

- [x] **Step 5: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_price_history_import.py -q`
Expected: PASS, 6 tests.

- [x] **Step 6: Commit**

```bash
git add backend/collectors/price_history_sources/ backend/tests/test_price_history_import.py
git commit -m "feat: add the cs2-prices-tracker price-history adapter"
```

---

### Task 2: Stall detection

**Files:**
- Create: `backend/collectors/price_history_import.py`
- Modify: `backend/tests/test_price_history_import.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `StalledSourceError(Exception)`, `detect_stalled_days(day_digests: dict[date, str]) -> list[list[date]]`.

- [x] **Step 1: Write the failing tests**

Append to `backend/tests/test_price_history_import.py`:

```python
from collectors.price_history_import import (
    StalledSourceError,
    detect_stalled_days,
)


def test_detect_stalled_days_finds_consecutive_identical_files():
    digests = {
        date(2026, 7, 26): "aaa",
        date(2026, 7, 27): "bbb",
        date(2026, 7, 28): "bbb",
        date(2026, 7, 29): "bbb",
        date(2026, 7, 30): "ccc",
    }
    assert detect_stalled_days(digests) == [
        [date(2026, 7, 27), date(2026, 7, 28), date(2026, 7, 29)]
    ]


def test_detect_stalled_days_ignores_identical_files_that_are_not_adjacent():
    digests = {
        date(2025, 6, 1): "aaa",
        date(2025, 6, 2): "bbb",
        date(2025, 6, 3): "aaa",
    }
    assert detect_stalled_days(digests) == []


def test_detect_stalled_days_is_empty_for_all_distinct_files():
    digests = {date(2025, 6, d): f"h{d}" for d in range(1, 6)}
    assert detect_stalled_days(digests) == []


def test_stalled_source_error_is_raisable_with_the_groups():
    with pytest.raises(StalledSourceError):
        raise StalledSourceError([[date(2026, 7, 27), date(2026, 7, 28)]])
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_price_history_import.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'collectors.price_history_import'`

- [x] **Step 3: Write the implementation**

Create `backend/collectors/price_history_import.py`:

```python
"""Shared machinery for one-shot historical price-source imports.

Source-specific fetching and parsing live in ``collectors/price_history_sources/``.
Everything here is source-agnostic and is what a second backfill source reuses.
"""
from datetime import date


class StalledSourceError(Exception):
    """Consecutive byte-identical day files: the upstream scraper stalled.

    This is fatal rather than a warning. A stalled scraper republishes the same
    prices under new dates, and a repeated price is not merely wrong — it
    becomes a frozen price RUN, which the label path treats as a real
    observation. `cs2-prices-tracker` is documented as stalling from roughly
    mid-July 2026, so a range that reaches into it must fail loudly.
    """

    def __init__(self, groups: list[list[date]]):
        self.groups = groups
        summary = "; ".join(
            f"{g[0].isoformat()}..{g[-1].isoformat()} ({len(g)} days)"
            for g in groups
        )
        super().__init__(
            f"upstream source stalled — byte-identical files across {summary}"
        )


def detect_stalled_days(day_digests: dict[date, str]) -> list[list[date]]:
    """Runs of >= 2 CONSECUTIVE days whose files hash identically.

    Adjacency is required. Two identical files a month apart are a coincidence
    of a quiet market; two in a row are a scraper that did not run.
    """
    groups: list[list[date]] = []
    run: list[date] = []
    previous_digest: str | None = None
    previous_day: date | None = None

    for day in sorted(day_digests):
        digest = day_digests[day]
        adjacent = previous_day is not None and (day - previous_day).days == 1
        if adjacent and digest == previous_digest:
            if not run:
                run = [previous_day]
            run.append(day)
        else:
            if len(run) >= 2:
                groups.append(run)
            run = []
        previous_digest, previous_day = digest, day

    if len(run) >= 2:
        groups.append(run)
    return groups
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_price_history_import.py -q`
Expected: PASS, 10 tests.

- [x] **Step 5: Commit**

```bash
git add backend/collectors/price_history_import.py backend/tests/test_price_history_import.py
git commit -m "feat: detect a stalled upstream source by consecutive identical files"
```

---

### Task 3: The gap quality gate

**Files:**
- Modify: `backend/collectors/price_history_import.py`
- Modify: `backend/tests/test_price_history_import.py`

**Interfaces:**
- Consumes: the record triple `(item_slug, day, price)` from Task 1.
- Produces: `MAX_GAP_DAYS: int`, `MIN_DISTINCT_DAYS: int`, `apply_gap_gate(records, max_gap_days=MAX_GAP_DAYS, min_distinct_days=MIN_DISTINCT_DAYS) -> tuple[list[tuple[str, date, float]], GapGateReport]`, and `GapGateReport` (a frozen dataclass with `kept_items: int`, `rejected_gap_items: int`, `rejected_sparse_items: int`, `kept_rows: int`, `rejected_rows: int`, `worst_gap_days: int`).

- [x] **Step 1: Write the failing tests**

Append to `backend/tests/test_price_history_import.py`:

```python
from datetime import timedelta

from collectors.price_history_import import (
    MAX_GAP_DAYS,
    MIN_DISTINCT_DAYS,
    apply_gap_gate,
)


def _series(name, days, price=1.0):
    """Records for *name* on the given day offsets from 2025-06-01."""
    return [(name, date(2025, 6, 1) + timedelta(days=d), price) for d in days]


# The gap-specific tests set min_distinct_days=1 so they isolate the gap
# condition; the sparse floor gets its own tests below.
def _gaps_only(records):
    return apply_gap_gate(records, min_distinct_days=1)


def test_max_gap_matches_the_archive_window_constant():
    from backtest.price_resolution import MAX_WINDOW_SPAN_DAYS
    assert MAX_GAP_DAYS == MAX_WINDOW_SPAN_DAYS == 7


def test_min_distinct_days_floor_is_180():
    assert MIN_DISTINCT_DAYS == 180


def test_a_seven_day_gap_is_kept():
    kept, report = _gaps_only(_series("Item A", [0, 7, 14]))
    assert {r[0] for r in kept} == {"Item A"}
    assert report.rejected_gap_items == 0


def test_an_eight_day_gap_is_rejected():
    kept, report = _gaps_only(_series("Item A", [0, 8, 16]))
    assert kept == []
    assert report.rejected_gap_items == 1
    assert report.rejected_rows == 3
    assert report.worst_gap_days == 8


def test_the_gate_is_per_item_not_global():
    records = _series("Dense", [0, 1, 2]) + _series("Sparse", [0, 30])
    kept, report = _gaps_only(records)
    assert {r[0] for r in kept} == {"Dense"}
    assert report.kept_items == 1
    assert report.rejected_gap_items == 1


def test_edges_are_not_penalised_only_interior_gaps_count():
    # Starts late and ends early inside the range; interior spacing is daily.
    kept, _ = _gaps_only(_series("Late Starter", [40, 41, 42, 43]))
    assert {r[0] for r in kept} == {"Late Starter"}


def test_a_dense_item_spanning_180_days_passes_the_default_floor():
    kept, report = apply_gap_gate(_series("Dense", range(180)))
    assert {r[0] for r in kept} == {"Dense"}
    assert report.rejected_sparse_items == 0


def test_179_distinct_days_is_one_short_and_is_rejected():
    kept, report = apply_gap_gate(_series("Nearly", range(179)))
    assert kept == []
    assert report.rejected_sparse_items == 1


def test_two_adjacent_observations_do_not_pass_trivially():
    """Max-gap alone is necessary and NOT sufficient.

    Two consecutive days have a max interior gap of 1, so without the distinct-
    day floor this item sails through and flips is_backfilled on two rows —
    measured at 904 such items (611 at >=$1) on the 90-day block.
    """
    kept, report = apply_gap_gate(_series("Two Days", [0, 1]))
    assert kept == []
    assert report.rejected_sparse_items == 1
    assert report.rejected_gap_items == 0


def test_a_single_observation_item_is_rejected_by_the_sparse_floor():
    kept, report = apply_gap_gate(_series("Lonely", [5]))
    assert kept == []
    assert report.rejected_sparse_items == 1


def test_the_gap_condition_takes_precedence_in_the_report():
    """An item failing BOTH conditions counts once, as a gap rejection."""
    kept, report = apply_gap_gate(_series("Both", [0, 30]))
    assert kept == []
    assert report.rejected_gap_items == 1
    assert report.rejected_sparse_items == 0


def test_records_are_deduplicated_on_item_and_day_keeping_the_first():
    records = [
        ("Item A", date(2025, 6, 1), 10.0),
        ("Item A", date(2025, 6, 1), 99.0),
        ("Item A", date(2025, 6, 2), 11.0),
    ]
    kept, report = _gaps_only(records)
    assert sorted(kept) == [
        ("Item A", date(2025, 6, 1), 10.0),
        ("Item A", date(2025, 6, 2), 11.0),
    ]
    assert report.kept_rows == 2
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_price_history_import.py -q`
Expected: FAIL — `ImportError: cannot import name 'MAX_GAP_DAYS'`

- [x] **Step 3: Write the implementation**

Add to `backend/collectors/price_history_import.py` (imports at the top of the file, rest appended):

```python
from collections import defaultdict
from dataclasses import dataclass

from backtest.price_resolution import MAX_WINDOW_SPAN_DAYS

#: An item whose observations are further apart than this yields a punctured
#: series: lags compute over holes and the resolution window breaks rather than
#: continuing. Deliberately the archive's own constant, not a new number.
MAX_GAP_DAYS = MAX_WINDOW_SPAN_DAYS

#: Minimum distinct days an item must carry inside the imported range.
#:
#: NOT redundant with MAX_GAP_DAYS, which passes trivially for a near-empty
#: item: two consecutive observations have a max interior gap of 1. Measured on
#: the 90-day block, 904 items (611 at >= $1) clear the gap bar on fewer than 10
#: of 90 days, median 2 — each of which would flip ``is_backfilled`` on two
#: rows. Together the two conditions mean "densely observed over at least half a
#: year". A genuinely dense item carries 300+ of the range's 408 days, so this
#: never binds on real data.
MIN_DISTINCT_DAYS = 180


@dataclass(frozen=True)
class GapGateReport:
    kept_items: int
    rejected_gap_items: int
    rejected_sparse_items: int
    kept_rows: int
    rejected_rows: int
    worst_gap_days: int


def apply_gap_gate(
    records: list[tuple[str, date, float]],
    max_gap_days: int = MAX_GAP_DAYS,
    min_distinct_days: int = MIN_DISTINCT_DAYS,
) -> tuple[list[tuple[str, date, float]], GapGateReport]:
    """Keep only densely-observed items: both conditions must hold.

    (a) no interior gap wider than *max_gap_days*, and
    (b) at least *min_distinct_days* distinct days.

    Filtering here rather than at training is deliberate. A sparse item still
    flips ``is_backfilled`` (derived as "has any row before 2026-01-01"), so
    admitting one buys a gate entry that carries no usable features — which is
    how that flag was rendered meaningless once before, when it read 8,691 of
    8,691. Price, by contrast, is NOT filtered here: which cohort to train on
    belongs to ``TRAIN_MIN_MEDIAN_PRICE``, and the archive must not bake it in.

    Only gaps BETWEEN consecutive observations count. An item that starts late
    or stops early is judged on the span it covers. An item failing both
    conditions is counted once, as a gap rejection.
    """
    by_item: dict[str, dict[date, float]] = defaultdict(dict)
    for slug, day, price in records:
        by_item[slug].setdefault(day, price)

    kept: list[tuple[str, date, float]] = []
    kept_items = rejected_gap = rejected_sparse = rejected_rows = 0
    worst_gap = 0

    for slug, observations in by_item.items():
        days = sorted(observations)
        gaps = [(b - a).days for a, b in zip(days, days[1:])]
        item_worst = max(gaps) if gaps else 0

        if item_worst > max_gap_days:
            rejected_gap += 1
            rejected_rows += len(days)
            worst_gap = max(worst_gap, item_worst)
            continue
        if len(days) < min_distinct_days:
            rejected_sparse += 1
            rejected_rows += len(days)
            continue

        kept_items += 1
        kept.extend((slug, day, observations[day]) for day in days)

    return kept, GapGateReport(
        kept_items=kept_items,
        rejected_gap_items=rejected_gap,
        rejected_sparse_items=rejected_sparse,
        kept_rows=len(kept),
        rejected_rows=rejected_rows,
        worst_gap_days=worst_gap,
    )
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_price_history_import.py -q`
Expected: PASS, 17 tests.

- [x] **Step 5: Commit**

```bash
git add backend/collectors/price_history_import.py backend/tests/test_price_history_import.py
git commit -m "feat: gate imported items on max interior gap, not on price"
```

---

### Task 4: Archive frame and write

**Files:**
- Modify: `backend/collectors/price_history_import.py`
- Modify: `backend/tests/test_price_history_import.py`

**Interfaces:**
- Consumes: the record triple from Task 1; `db.parquet.append_monthly(out_dir, prefix, df, dedup_keys, day_col="day")`; `db.archive.prices_relation(con, archive_dir=..., columns=..., where=...)`.
- Produces: `to_archive_frame(records, source: str, ingested_at: datetime) -> pd.DataFrame` and `write_archive_frame(frame, out_dir) -> int`.

- [x] **Step 1: Write the failing tests**

Append to `backend/tests/test_price_history_import.py`:

```python
import duckdb
import pandas as pd

from collectors.price_history_import import to_archive_frame, write_archive_frame
from db.archive import CANONICAL_PRICE_COLUMNS, prices_relation

_INGESTED = datetime(2026, 8, 8, 12, 0, 0)


def test_archive_frame_has_exactly_the_canonical_columns():
    frame = to_archive_frame(
        [("AK-47 | Redline (Field-Tested)", date(2025, 6, 1), 12.5)],
        source="tracker_steam_24h",
        ingested_at=_INGESTED,
    )
    assert list(frame.columns) == list(CANONICAL_PRICE_COLUMNS)


def test_archive_frame_leaves_volume_null_never_zero():
    frame = to_archive_frame(
        [("AK-47 | Redline (Field-Tested)", date(2025, 6, 1), 12.5)],
        source="tracker_steam_24h",
        ingested_at=_INGESTED,
    )
    assert frame["volume"].isna().all()
    assert not (frame["volume"].fillna(-1) == 0).any()


def test_archive_frame_stamps_arrival_not_the_day_it_describes():
    frame = to_archive_frame(
        [("AK-47 | Redline (Field-Tested)", date(2025, 6, 1), 12.5)],
        source="tracker_steam_24h",
        ingested_at=_INGESTED,
    )
    assert frame["ingested_at"].iloc[0] == pd.Timestamp(_INGESTED)
    assert frame["day"].iloc[0] != frame["ingested_at"].iloc[0]


def test_written_rows_read_back_through_the_typed_reader(tmp_path):
    frame = to_archive_frame(
        [
            ("AK-47 | Redline (Field-Tested)", date(2025, 6, 1), 12.5),
            ("AK-47 | Redline (Field-Tested)", date(2026, 3, 1), 13.5),
        ],
        source="tracker_steam_24h",
        ingested_at=_INGESTED,
    )
    out_dir = tmp_path / "price-archive"
    assert write_archive_frame(frame, out_dir) == 2

    con = duckdb.connect()
    try:
        rel = prices_relation(
            con,
            archive_dir=out_dir,
            columns=["item_slug", "day", "source", "mean_price", "volume"],
        )
        rows = con.sql(
            f"SELECT item_slug, source, mean_price, volume FROM {rel} ORDER BY day"
        ).fetchall()
    finally:
        con.close()

    assert rows == [
        ("AK-47 | Redline (Field-Tested)", "tracker_steam_24h", 12.5, None),
        ("AK-47 | Redline (Field-Tested)", "tracker_steam_24h", 13.5, None),
    ]


def test_rows_fan_out_to_one_file_per_month(tmp_path):
    frame = to_archive_frame(
        [
            ("Item A", date(2025, 6, 1), 1.0),
            ("Item A", date(2025, 7, 1), 1.0),
        ],
        source="tracker_steam_24h",
        ingested_at=_INGESTED,
    )
    out_dir = tmp_path / "price-archive"
    write_archive_frame(frame, out_dir)
    names = sorted(p.name for p in out_dir.glob("prices-*.parquet"))
    assert names == ["prices-2025-06.parquet", "prices-2025-07.parquet"]


def test_a_reappend_keeps_the_first_arrival_not_the_latest(tmp_path):
    """append_monthly replaces a colliding row wholesale, ingested_at included.

    Without _preserve_first_arrival a re-run dates every touched row forward,
    which is exactly what the embargo reads. The daily CI writer solves the
    same problem at scripts/append_to_parquet.py:267-270.
    """
    out_dir = tmp_path / "price-archive"
    first = datetime(2026, 8, 8, 12, 0, 0)
    later = datetime(2026, 9, 1, 9, 30, 0)
    records = [("Item A", date(2025, 6, 1), 10.0)]

    write_archive_frame(to_archive_frame(records, "tracker_steam_24h", first), out_dir)
    write_archive_frame(to_archive_frame(records, "tracker_steam_24h", later), out_dir)

    con = duckdb.connect()
    try:
        rel = prices_relation(
            con, archive_dir=out_dir, columns=["item_slug", "ingested_at"]
        )
        stored = con.sql(f"SELECT ingested_at FROM {rel}").fetchall()
    finally:
        con.close()

    assert len(stored) == 1
    assert stored[0][0] == first


def test_a_row_with_no_prior_arrival_takes_the_new_timestamp(tmp_path):
    """min skips NaT: a row predating the column must not stay unknown."""
    out_dir = tmp_path / "price-archive"
    stamped = to_archive_frame(
        [("Item A", date(2025, 6, 1), 10.0)], "tracker_steam_24h", _INGESTED
    )
    unstamped = stamped.copy()
    unstamped["ingested_at"] = pd.NaT
    write_archive_frame(unstamped, out_dir)
    write_archive_frame(stamped, out_dir)

    con = duckdb.connect()
    try:
        rel = prices_relation(
            con, archive_dir=out_dir, columns=["item_slug", "ingested_at"]
        )
        stored = con.sql(f"SELECT ingested_at FROM {rel}").fetchall()
    finally:
        con.close()

    assert stored[0][0] == pd.Timestamp(_INGESTED)


def test_a_reappend_does_not_duplicate_the_same_item_day_source(tmp_path):
    frame = to_archive_frame(
        [("Item A", date(2025, 6, 1), 1.0)],
        source="tracker_steam_24h",
        ingested_at=_INGESTED,
    )
    out_dir = tmp_path / "price-archive"
    write_archive_frame(frame, out_dir)
    write_archive_frame(frame, out_dir)

    con = duckdb.connect()
    try:
        rel = prices_relation(con, archive_dir=out_dir, columns=["item_slug"])
        count = con.sql(f"SELECT count(*) FROM {rel}").fetchone()[0]
    finally:
        con.close()
    assert count == 1
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_price_history_import.py -q`
Expected: FAIL — `ImportError: cannot import name 'to_archive_frame'`

- [x] **Step 3: Write the implementation**

Add to `backend/collectors/price_history_import.py`:

```python
from datetime import datetime
from pathlib import Path

import pandas as pd

from db.archive import CANONICAL_PRICE_COLUMNS
from db.parquet import append_monthly

#: The archive's natural key. A re-append REPLACES the row: `_append_parquet`
#: selects `_new` unconditionally and keeps an existing row only `WHERE NOT
#: EXISTS` a match (`db/parquet.py:250-262`), so the new row's `ingested_at`
#: wins too.
#:
#: That is wrong for an arrival timestamp, which is why `write_archive_frame`
#: restores first-arrival itself. The daily CI writer has the same problem and
#: solves it the same way (`scripts/append_to_parquet.py:267-270`); the
#: difference is that it does so before its own dedup, while `append_monthly`
#: offers no hook, so the correction has to happen on the frame going in.
DEDUP_KEYS = ["item_slug", "day", "source"]


def to_archive_frame(
    records: list[tuple[str, date, float]],
    source: str,
    ingested_at: datetime,
) -> pd.DataFrame:
    """Canonical price rows for *records*, labelled *source*.

    ``volume`` is a typed NULL, never 0. No aggregator feed carries volume and
    a real zero never occurs — a day with no sale produces an ABSENT row. The
    fabricated zeros are what kept ``has_volume`` reading True and
    ``volume_missing`` reporting "present", which shelved eleven features.

    ``ingested_at`` is the run's wall clock: when the row ARRIVED, not what it
    describes. A backfill writer violates "a row dated `d` was knowable on `d`"
    by definition, which is precisely why the column exists.
    """
    frame = pd.DataFrame(records, columns=["item_slug", "day", "mean_price"])
    frame["day"] = pd.to_datetime(frame["day"]).dt.date
    frame["source"] = source
    frame["volume"] = pd.Series([pd.NA] * len(frame), dtype="Int64")
    frame["ingested_at"] = pd.Timestamp(ingested_at)
    return frame[list(CANONICAL_PRICE_COLUMNS)]


def _preserve_first_arrival(
    frame: pd.DataFrame, out_dir: Path | str
) -> pd.DataFrame:
    """Roll each row's ``ingested_at`` back to its earliest recorded arrival.

    ``append_monthly`` replaces a colliding row wholesale, so without this a
    re-run re-stamps arrival forward on every row it touches — and the whole
    reason the column exists is the embargo. A value being corrected still
    became knowable on the day it first landed.

    ``min`` skips NaT, so a row predating the column takes the new timestamp
    rather than staying unknown.
    """
    files = sorted(Path(out_dir).glob("prices-*.parquet"))
    if not files:
        return frame

    con = duckdb.connect()
    try:
        rel = prices_relation(
            con,
            archive_dir=Path(out_dir),
            columns=["item_slug", "day", "source", "ingested_at"],
        )
        existing = con.sql(
            f"SELECT item_slug, day, source, ingested_at FROM {rel}"
        ).fetchdf()
    finally:
        con.close()

    if existing.empty:
        return frame

    existing["day"] = pd.to_datetime(existing["day"]).dt.date
    merged = frame.merge(
        existing.rename(columns={"ingested_at": "_prior"}),
        on=DEDUP_KEYS,
        how="left",
    )
    frame = frame.copy()
    frame["ingested_at"] = merged[["ingested_at", "_prior"]].min(axis=1)
    return frame


def write_archive_frame(frame: pd.DataFrame, out_dir: Path | str) -> int:
    """Append *frame* to the monthly price files under *out_dir*.

    Returns the row count written. *out_dir* is the ``price-archive`` directory
    itself, so a caller passing a staging root must append that component.
    """
    if frame.empty:
        return 0
    append_monthly(out_dir, "prices", _preserve_first_arrival(frame, out_dir),
                   DEDUP_KEYS)
    return len(frame)
```

This needs `import duckdb` and `from db.archive import prices_relation` alongside the existing imports.

- [x] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_price_history_import.py -q`
Expected: PASS, 23 tests.

- [x] **Step 5: Commit**

```bash
git add backend/collectors/price_history_import.py backend/tests/test_price_history_import.py
git commit -m "feat: build and write source-labelled archive rows with NULL volume"
```

---

### Task 5: The CLI — fetch, cache, orchestrate

**Files:**
- Create: `backend/scripts/import_price_history_source.py`
- Modify: `backend/tests/test_price_history_import.py`

**Interfaces:**
- Consumes: `ADAPTERS` (Task 1); `detect_stalled_days`, `StalledSourceError` (Task 2); `apply_gap_gate` (Task 3); `to_archive_frame`, `write_archive_frame` (Task 4).
- Produces: `daterange(start: date, end: date) -> list[date]`, `cached_path(cache_dir: Path, source_name: str, day: date) -> Path`, `load_cached_day(path: Path) -> dict | None`, and `main(argv: list[str] | None = None) -> int`.

- [x] **Step 1: Write the failing tests**

Append to `backend/tests/test_price_history_import.py`:

```python
import json

from scripts.import_price_history_source import (
    cached_path,
    daterange,
    load_cached_day,
)


def test_daterange_is_inclusive_of_both_ends():
    days = daterange(date(2025, 6, 1), date(2025, 6, 4))
    assert days == [
        date(2025, 6, 1), date(2025, 6, 2), date(2025, 6, 3), date(2025, 6, 4)
    ]


def test_daterange_rejects_an_inverted_range():
    with pytest.raises(ValueError):
        daterange(date(2025, 6, 4), date(2025, 6, 1))


def test_cached_path_is_namespaced_by_source(tmp_path):
    path = cached_path(tmp_path, "cs2_prices_tracker", date(2025, 6, 1))
    assert path == tmp_path / "cs2_prices_tracker" / "2025-06-01.json"


def test_load_cached_day_returns_none_for_a_missing_file(tmp_path):
    assert load_cached_day(tmp_path / "nope.json") is None


def test_load_cached_day_returns_none_for_an_empty_or_corrupt_file(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text("")
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json")
    assert load_cached_day(empty) is None
    assert load_cached_day(corrupt) is None


def test_load_cached_day_parses_a_good_file(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"Item A": {"steam": {"last_24h": 1.0}}}))
    assert load_cached_day(good) == {"Item A": {"steam": {"last_24h": 1.0}}}
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_price_history_import.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.import_price_history_source'`

- [x] **Step 3: Write the CLI**

Create `backend/scripts/import_price_history_source.py`:

```python
#!/usr/bin/env python3
"""One-shot import of a historical price source into a STAGING archive.

Usage (from ``backend/``)::

    venv/bin/python scripts/import_price_history_source.py \
        --source cs2_prices_tracker \
        --start 2025-02-17 --end 2026-03-31 \
        --cache-dir runtime/price_history_cache \
        --out-dir ../archive-staging

Writes ``{out-dir}/price-archive/prices-YYYY-MM.parquet``. It never touches
``price-archive/`` and never pushes to ``cs2-oracle-data`` — promotion is a
separate, deliberate step.

The fetch is resumable: a cached day that parses is not re-requested, so an
interrupted run resumes without re-downloading ~2.3 GB.

`backend/.env` points at production Supabase, but nothing here imports
``database``, so this script holds no DB connection.
"""
import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors.price_history_import import (  # noqa: E402
    MIN_DISTINCT_DAYS,
    StalledSourceError,
    apply_gap_gate,
    detect_stalled_days,
    to_archive_frame,
    write_archive_frame,
)
from collectors.price_history_sources import ADAPTERS  # noqa: E402

logger = logging.getLogger(__name__)

REQUEST_DELAY = 0.2
REQUEST_TIMEOUT = 60


def daterange(start: date, end: date) -> list[date]:
    """Every day from *start* to *end*, both inclusive."""
    if end < start:
        raise ValueError(f"end {end} precedes start {start}")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def cached_path(cache_dir: Path, source_name: str, day: date) -> Path:
    """Where one fetched day file lives, namespaced by source."""
    return Path(cache_dir) / source_name / f"{day.isoformat()}.json"


def load_cached_day(path: Path) -> dict | None:
    """The parsed payload, or None if absent, empty or unparseable.

    Returning None rather than raising is what makes the run resumable: a
    truncated file from an interrupted download is simply re-fetched.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def fetch_days(adapter, source_name, days, cache_dir, session):
    """Populate the cache for *days*. Returns {day: md5} for what is on disk."""
    digests: dict[date, str] = {}
    fetched = skipped = missing = 0

    for day in days:
        path = cached_path(cache_dir, source_name, day)
        path.parent.mkdir(parents=True, exist_ok=True)

        if load_cached_day(path) is None:
            response = session.get(adapter.day_url(day), timeout=REQUEST_TIMEOUT)
            if response.status_code == 404:
                missing += 1
                logger.warning(f"  {day}: absent upstream (404)")
                continue
            response.raise_for_status()
            path.write_bytes(response.content)
            fetched += 1
            time.sleep(REQUEST_DELAY)
        else:
            skipped += 1

        digests[day] = hashlib.md5(path.read_bytes()).hexdigest()

    logger.info(
        f"  fetched {fetched:,}, cached {skipped:,}, absent upstream {missing:,}"
    )
    return digests


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--start", required=True, help="inclusive, YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="inclusive, YYYY-MM-DD")
    ap.add_argument("--cache-dir", default="runtime/price_history_cache")
    ap.add_argument("--out-dir", default="../archive-staging")
    ap.add_argument("--fetch-only", action="store_true",
                    help="populate the cache and stop, writing nothing")
    args = ap.parse_args(argv)

    adapter = ADAPTERS[args.source]
    days = daterange(
        date.fromisoformat(args.start), date.fromisoformat(args.end)
    )
    logger.info(
        f"{args.source}: {len(days):,} days, "
        f"{days[0].isoformat()} -> {days[-1].isoformat()}"
    )

    session = requests.Session()
    digests = fetch_days(adapter, args.source, days, Path(args.cache_dir), session)

    stalled = detect_stalled_days(digests)
    if stalled:
        raise StalledSourceError(stalled)

    if args.fetch_only:
        logger.info("--fetch-only: nothing written")
        return 0

    records: list[tuple[str, date, float]] = []
    for day in sorted(digests):
        payload = load_cached_day(cached_path(Path(args.cache_dir), args.source, day))
        records.extend(adapter.parse_day(payload, day))
    logger.info(f"parsed {len(records):,} raw records")

    kept, report = apply_gap_gate(records)
    logger.info(
        f"quality gate: kept {report.kept_items:,} items / "
        f"{report.kept_rows:,} rows; rejected "
        f"{report.rejected_gap_items:,} on gaps (worst "
        f"{report.worst_gap_days or 0}d) and "
        f"{report.rejected_sparse_items:,} on the {MIN_DISTINCT_DAYS}-day floor, "
        f"{report.rejected_rows:,} rows total"
    )

    frame = to_archive_frame(kept, adapter.SOURCE, datetime.now())
    out_dir = Path(args.out_dir) / "price-archive"
    written = write_archive_frame(frame, out_dir)
    logger.info(f"wrote {written:,} rows as source={adapter.SOURCE} to {out_dir}")

    pre_2026 = sum(1 for r in kept if r[1] < date(2026, 1, 1))
    gate_items = len({r[0] for r in kept if r[1] < date(2026, 1, 1)})
    logger.info(
        f"pre-2026 rows {pre_2026:,} across {gate_items:,} items "
        f"— these are what is_backfilled tests for"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_price_history_import.py -q`
Expected: PASS, 29 tests.

- [x] **Step 5: Verify the CLI wires up end to end without network**

Run: `venv/bin/python scripts/import_price_history_source.py --help`
Expected: usage text listing `--source {cs2_prices_tracker}`.

- [x] **Step 6: Commit**

```bash
git add backend/scripts/import_price_history_source.py backend/tests/test_price_history_import.py
git commit -m "feat: add the resumable staging importer CLI"
```

---

### Task 6: Run the import and report the promotion-gate numbers

**Files:**
- Modify: `backend/scripts/import_price_history_source.py` (add `--report`)
- Modify: `backend/tests/test_price_history_import.py`

**Interfaces:**
- Consumes: everything above; `db.archive.prices_relation`.
- Produces: `report_promotion_gate(staging_dir: Path, archive_dir: Path, source: str) -> dict` with keys `rows`, `items`, `pre_2026_items`, `new_gate_items`, `zero_volume_rows`, `duplicate_keys`, `overlap_items`, `overlap_ratio_median`, `overlap_ratio_p25`, `overlap_ratio_p75`.

`new_gate_items` is the spec's promotion-gate item 4 — staged items with pre-2026 rows that the real archive does **not** already have pre-2026 rows for. That, not `pre_2026_items`, is the `is_backfilled` flip count, because the source also covers 95.8% of items already inside the gate.

- [x] **Step 1: Write the failing test**

Append to `backend/tests/test_price_history_import.py`:

```python
from scripts.import_price_history_source import report_promotion_gate


def test_report_promotion_gate_counts_rows_items_and_finds_no_zero_volume(tmp_path):
    staging = tmp_path / "staging" / "price-archive"
    frame = to_archive_frame(
        [
            ("Item A", date(2025, 6, 1), 10.0),
            ("Item A", date(2026, 3, 1), 11.0),
            ("Item B", date(2026, 3, 1), 20.0),
        ],
        source="tracker_steam_24h",
        ingested_at=_INGESTED,
    )
    write_archive_frame(frame, staging)

    archive = tmp_path / "archive" / "price-archive"
    existing = to_archive_frame(
        [("Item A", date(2026, 3, 1), 10.0)],
        source="aggregator_sync",
        ingested_at=_INGESTED,
    )
    write_archive_frame(existing, archive)

    result = report_promotion_gate(staging, archive, "tracker_steam_24h")
    assert result["rows"] == 3
    assert result["items"] == 2
    assert result["pre_2026_items"] == 1
    # Item A has a pre-2026 staged row and the real archive has none for it,
    # so it flips. Item B is 2026-only and flips nothing.
    assert result["new_gate_items"] == 1
    assert result["zero_volume_rows"] == 0
    assert result["duplicate_keys"] == 0
    assert result["overlap_items"] == 1
    assert result["overlap_ratio_median"] == pytest.approx(1.1)


def test_report_promotion_gate_excludes_items_already_inside_the_gate(tmp_path):
    staging = tmp_path / "staging" / "price-archive"
    write_archive_frame(
        to_archive_frame(
            [("Already Gated", date(2025, 6, 1), 10.0)],
            source="tracker_steam_24h",
            ingested_at=_INGESTED,
        ),
        staging,
    )
    archive = tmp_path / "archive" / "price-archive"
    write_archive_frame(
        to_archive_frame(
            [("Already Gated", date(2024, 5, 1), 9.0)],
            source="aggregator_sync",
            ingested_at=_INGESTED,
        ),
        archive,
    )

    result = report_promotion_gate(staging, archive, "tracker_steam_24h")
    assert result["pre_2026_items"] == 1
    assert result["new_gate_items"] == 0
```

- [x] **Step 2: Run the test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_price_history_import.py -q`
Expected: FAIL — `ImportError: cannot import name 'report_promotion_gate'`

- [x] **Step 3: Implement the report**

Add to `backend/scripts/import_price_history_source.py`:

```python
import duckdb  # noqa: E402

from db.archive import prices_relation  # noqa: E402


def report_promotion_gate(staging_dir, archive_dir, source: str) -> dict:
    """The numbers the spec's promotion gate requires.

    ``overlap_ratio_*`` is the seam measurement: imported price over the
    existing archive's price for the same (item, day). This is Steam-to-Steam,
    unlike the earlier repo-vs-7-source-consensus figure, which overstated the
    disagreement by mixing in non-Steam venues.
    """
    con = duckdb.connect()
    try:
        staged = prices_relation(
            con,
            archive_dir=Path(staging_dir),
            columns=["item_slug", "day", "source", "mean_price", "volume"],
            where=f"source = '{source}'",
        )
        rows, items, pre_2026_items, zero_volume = con.sql(f"""
            SELECT count(*),
                   count(DISTINCT item_slug),
                   count(DISTINCT CASE WHEN day < DATE '2026-01-01'
                                       THEN item_slug END),
                   count(*) FILTER (WHERE volume = 0)
            FROM {staged}
        """).fetchone()

        duplicate_keys = con.sql(f"""
            SELECT count(*) FROM (
                SELECT item_slug, day, source
                FROM {staged}
                GROUP BY 1, 2, 3 HAVING count(*) > 1
            )
        """).fetchone()[0]

        existing = prices_relation(
            con,
            archive_dir=Path(archive_dir),
            columns=["item_slug", "day", "source", "mean_price"],
            where=f"source IS DISTINCT FROM '{source}'",
        )
        new_gate_items = con.sql(f"""
            SELECT count(*) FROM (
                SELECT DISTINCT item_slug FROM {staged}
                WHERE day < DATE '2026-01-01'
                EXCEPT
                SELECT DISTINCT item_slug FROM {existing}
                WHERE day < DATE '2026-01-01'
            )
        """).fetchone()[0]

        overlap = con.sql(f"""
            WITH consensus AS (
                SELECT item_slug, day, median(mean_price) AS price
                FROM {existing} GROUP BY 1, 2
            ), paired AS (
                SELECT s.item_slug,
                       s.mean_price / c.price AS ratio
                FROM {staged} s
                JOIN consensus c
                  ON c.item_slug = s.item_slug AND c.day = s.day
                WHERE c.price > 0
            )
            SELECT count(DISTINCT item_slug),
                   median(ratio),
                   quantile_cont(ratio, 0.25),
                   quantile_cont(ratio, 0.75)
            FROM paired
        """).fetchone()
    finally:
        con.close()

    return {
        "rows": rows,
        "items": items,
        "pre_2026_items": pre_2026_items,
        "new_gate_items": new_gate_items,
        "zero_volume_rows": zero_volume,
        "duplicate_keys": duplicate_keys,
        "overlap_items": overlap[0],
        "overlap_ratio_median": overlap[1],
        "overlap_ratio_p25": overlap[2],
        "overlap_ratio_p75": overlap[3],
    }
```

Add these two `add_argument` calls in `main`, alongside the existing ones and **before** `args = ap.parse_args(argv)`:

```python
    ap.add_argument("--report", action="store_true",
                    help="report the promotion-gate numbers for an existing "
                         "staging import and exit, fetching nothing")
    ap.add_argument("--archive-dir", default="../price-archive",
                    help="the real archive, compared against for the seam read")
```

Then, immediately after `adapter = ADAPTERS[args.source]` and before the `daterange` call, short-circuit:

```python
    if args.report:
        result = report_promotion_gate(
            Path(args.out_dir) / "price-archive",
            Path(args.archive_dir),
            adapter.SOURCE,
        )
        for key, value in result.items():
            logger.info(f"  {key}: {value}")
        return 0
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_price_history_import.py -q`
Expected: PASS, 30 tests.

- [x] **Step 5: Run the real import**

```bash
venv/bin/python scripts/import_price_history_source.py \
    --source cs2_prices_tracker \
    --start 2025-02-17 --end 2026-03-31 \
    --cache-dir runtime/price_history_cache \
    --out-dir ../archive-staging
```

Expected: ~408 days fetched (~2.3 GB cached), no `StalledSourceError`, quality-gate rejections logged split by cause, rows written to `../archive-staging/price-archive/`.

Projection from the spec, to compare against: **~10,500 items gaining pre-2026 rows** (unfiltered is 18,843; the max-gap condition passes ~64% and the 180-day floor removes a further ~13%). If the log's `pre_2026 items` figure falls outside **9,000–12,000**, stop and reconcile before going further — the band is wide because both factors are extrapolations from a 90-day block, not measurements over the 408-day range.

- [x] **Step 6: Report the promotion-gate numbers**

```bash
venv/bin/python scripts/import_price_history_source.py \
    --source cs2_prices_tracker --start 2025-02-17 --end 2026-03-31 \
    --out-dir ../archive-staging --archive-dir ../price-archive --report
```

Expected: `zero_volume_rows: 0`, `duplicate_keys: 0`, and an `overlap_ratio_median` near 1.0. Record all nine numbers — they are the promotion decision.

- [x] **Step 7: Run the full backend suite**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: no new failures against the pre-change baseline (1,233 passing as of 2026-08-08).

- [x] **Step 8: Commit**

```bash
git add backend/scripts/import_price_history_source.py backend/tests/test_price_history_import.py
git commit -m "feat: report the promotion-gate numbers for a staged import"
```

---

---

### Task 7: Fee-correct the source and floor it at $1

Added 2026-08-09 after the Task 6 run. The promotion gate refuted this plan's stated premise that the source needs no basis conversion.

**Measured:** `tracker_steam_24h` ÷ `aggregator_sync` = **1.147–1.154, flat across all seven price tiers** (1.107 at <$0.10 through 1.163 at ≥$100), median within-item CV 0.046–0.063. Flat is the signature of a constant, not a market wedge. Dividing by **1.1607** collapses it to **0.998** at ≥$1. The source serves Steam's **buyer** price (fee included); the archive stores **net**. `scripts/backfill_steam_listing_history.py` already carries `STEAM_FEE_MULTIPLIER` for exactly this.

**Why the $1 floor, and why it does not contradict this plan's "no price floor at import" rule.** That rule exists so the archive does not bake in a *cohort* decision. This floor is a *data-validity* decision: the fee constant is documented as synthetic — flat where fee theory demands ~1.67 at $0.03 falling to ~1.15 at $50 — and the correction measures 0.998 at ≥$1 but only 0.906 below it. Sub-$1 rows would be knowingly biased. Cohort selection still belongs to `TRAIN_MIN_MEDIAN_PRICE`.

**Order matters:** correct the fee first, then apply the floor, so the floor is evaluated on net prices.

**Files:**
- Modify: `backend/collectors/price_history_sources/cs2_prices_tracker.py`
- Modify: `backend/collectors/price_history_import.py`
- Test: `backend/tests/test_price_history_import.py`

**Interfaces:**
- Consumes: `parse_day`, `apply_gap_gate` as built.
- Produces: `cs2_prices_tracker.STEAM_FEE_MULTIPLIER: float`; `apply_gap_gate(..., min_median_price: float | None = None)`; `GapGateReport` gains `rejected_cheap_items: int`.

- [x] **Step 1: Write the failing tests**

```python
def test_parse_day_returns_net_of_the_steam_fee():
    """The source serves the BUYER price; the archive stores NET.

    Measured 2026-08-09: tracker/aggregator_sync = 1.147-1.154 flat across all
    seven price tiers, collapsing to 0.998 at >=$1 after this division.
    """
    payload = {"AK-47 | Redline (Field-Tested)": {"steam": {"last_24h": 11.607}}}
    records = tracker.parse_day(payload, date(2025, 6, 15))
    assert records[0][2] == pytest.approx(10.0, rel=1e-6)


def test_steam_fee_multiplier_matches_the_repo_constant():
    from scripts.backfill_steam_listing_history import STEAM_FEE_MULTIPLIER
    assert tracker.STEAM_FEE_MULTIPLIER == STEAM_FEE_MULTIPLIER == 1.1607


def test_a_price_that_is_positive_only_before_the_fee_still_survives():
    payload = {"Cheap": {"steam": {"last_24h": 0.02}}}
    records = tracker.parse_day(payload, date(2025, 6, 15))
    assert len(records) == 1
    assert records[0][2] == pytest.approx(0.02 / 1.1607)


def test_the_gate_drops_items_below_the_median_price_floor():
    cheap = _series("Cheap", range(200), price=0.50)
    rich = _series("Rich", range(200), price=5.00)
    kept, report = apply_gap_gate(cheap + rich, min_median_price=1.0)
    assert {r[0] for r in kept} == {"Rich"}
    assert report.rejected_cheap_items == 1


def test_the_floor_uses_the_median_not_the_last_price():
    """One spike must not carry an otherwise-cheap item over the floor."""
    records = _series("Spiky", range(199), price=0.50) + [
        ("Spiky", date(2025, 6, 1) + timedelta(days=199), 500.0)
    ]
    kept, report = apply_gap_gate(records, min_median_price=1.0)
    assert kept == []
    assert report.rejected_cheap_items == 1


def test_no_floor_by_default_keeps_the_archive_free_of_a_cohort_decision():
    kept, report = apply_gap_gate(_series("Cheap", range(200), price=0.50))
    assert {r[0] for r in kept} == {"Cheap"}
    assert report.rejected_cheap_items == 0
```

- [x] **Step 2: Run to verify they fail**

Run: `venv/bin/python -m pytest tests/test_price_history_import.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'STEAM_FEE_MULTIPLIER'`

- [x] **Step 3: Implement**

In `cs2_prices_tracker.py`, add the constant and divide in `parse_day`:

```python
#: Steam's listed price is GROSS of its fee; the archive stores NET. Measured
#: 2026-08-09 against `aggregator_sync`: the raw ratio is 1.147-1.154 and FLAT
#: across seven price tiers, which is the signature of a constant rather than a
#: market wedge, and dividing by this collapses it to 0.998 at >= $1.
#:
#: The constant is documented as synthetic and is NOT trustworthy below ~$1
#: (the correction lands at 0.906 there), which is why the import applies a $1
#: floor. Kept equal to the repo's existing value rather than re-derived, so
#: there is one number to fix if it is ever re-measured.
STEAM_FEE_MULTIPLIER = 1.1607
```

and in `parse_day`, after the `price <= 0` check:

```python
        records.append((name, day, price / STEAM_FEE_MULTIPLIER))
```

In `price_history_import.py`, add `rejected_cheap_items: int` to `GapGateReport`, add the parameter, and apply it after the sparse check:

```python
        if min_median_price is not None:
            prices = sorted(observations[day] for day in days)
            mid = len(prices) // 2
            median_price = (prices[mid] if len(prices) % 2
                            else (prices[mid - 1] + prices[mid]) / 2)
            if median_price < min_median_price:
                rejected_cheap += 1
                rejected_rows += len(days)
                continue
```

- [x] **Step 4: Run to verify they pass**

Run: `venv/bin/python -m pytest tests/test_price_history_import.py -q`
Expected: PASS, 44 tests.

- [x] **Step 5: Wire the floor into the CLI**

Add `--min-median-price` (type float, default `1.0`) to `main`'s argparse, pass it into `apply_gap_gate`, and extend the gate log line with `rejected_cheap_items`.

- [x] **Step 6: Commit**

```bash
git add backend/collectors/ backend/tests/test_price_history_import.py
git commit -m "fix: import Steam prices net of fee, and floor the source at \$1"
```

## Not in this plan

Per the spec, all deliberately out of scope: promoting staging into `cs2-oracle-data`; retraining or changing `TRAIN_FEATURE_ROWS` / `TRAIN_MIN_MEDIAN_PRICE`; changing the `is_backfilled` derivation; raising `MIN_SERVED_PRICE_USD`; repairing the `volume` column; building a second adapter. The `database.py` note recording the gate's widened meaning is a promotion-time change, not an import-time one.
