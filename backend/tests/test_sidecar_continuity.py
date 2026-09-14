"""Tests for the sidecar continuity audit (Phase 1 gate)."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from scripts.check_sidecar_continuity import audit  # noqa: E402


def _supply_day(tmp_path: Path, day: date, n: int = 3):
    rows = pd.DataFrame(
        {
            "item_slug": [f"item-{i}" for i in range(n)],
            "snapshot_day": [day] * n,
            "source": ["waxpeer"] * n,
            "listing_count": [10] * n,
        }
    )
    path = tmp_path / f"supply-{day:%Y-%m}.parquet"
    if path.exists():
        prev = pd.read_parquet(path)
        rows = pd.concat([prev, rows], ignore_index=True)
    rows.to_parquet(path, index=False)


def test_reports_days_gaps_and_runs(tmp_path):
    d0 = date(2026, 9, 1)
    for d in (d0, d0 + timedelta(days=1), d0 + timedelta(days=3)):
        _supply_day(tmp_path, d)
    rep = audit(tmp_path)["supply"]
    assert rep["days"] == 3
    assert str(d0 + timedelta(days=2)) in rep["gaps"]
    assert rep["longest_run"] == 2
    assert rep["latest"] == str(d0 + timedelta(days=3))
    assert rep["per_source_latest"] == {"waxpeer": 3}


def test_missing_table_is_zero_not_error(tmp_path):
    rep = audit(tmp_path)["reddit-events"]
    assert rep["days"] == 0
    assert rep["latest"] is None
