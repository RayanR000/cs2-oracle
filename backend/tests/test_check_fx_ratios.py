"""check_fx_ratios: a CNY conversion error moves both CNY venues against csfloat, and nothing else.

The monitor must stay quiet on a stable archive, flag a single CNY venue's jump as a breach,
call it an FX signature only when both CNY venues move together and the USD controls do not,
and not mistake csfloat (the denominator) moving for an FX error.
"""

from __future__ import annotations

import datetime as dt
import json

import pyarrow as pa
import pyarrow.parquet as pq
from scripts.check_fx_ratios import CNY_VENUES, MIN_ITEMS, REFERENCE, USD_CONTROLS, daily_ratios, evaluate, main

DAYS = [dt.date(2026, 9, 1) + dt.timedelta(days=i) for i in range(16)]
LEVEL = {
    REFERENCE: 1.0,
    "aggregator_buff163": 1.017,
    "aggregator_youpin": 1.027,
    "aggregator_skinport": 1.12,
    "aggregator_csmoney": 1.04,
}


def _archive(tmp_path, shock=None):
    """MIN_ITEMS + 10 items on every venue; `shock` scales {venue: factor} on the last day."""
    rows = []
    for day in DAYS:
        for i in range(MIN_ITEMS + 10):
            base = 2.0 + i * 0.01
            for venue, level in LEVEL.items():
                f = (shock or {}).get(venue, 1.0) if day == DAYS[-1] else 1.0
                rows.append((f"item {i:04d}", day, venue, base * level * f, 1))
    cols = list(zip(*rows, strict=True))
    pq.write_table(
        pa.table(
            {
                "item_slug": pa.array(cols[0], type=pa.string()),
                "day": pa.array(cols[1], type=pa.date32()),
                "source": pa.array(cols[2], type=pa.string()),
                "mean_price": pa.array(cols[3], type=pa.float64()),
                "volume": pa.array(cols[4], type=pa.int64()),
            }
        ),
        tmp_path / "prices-2026-09.parquet",
    )
    return tmp_path


def _status(tmp_path, shock=None):
    return evaluate(daily_ratios(_archive(tmp_path, shock), 21), 14, 0.03)


def test_stable_archive_is_ok(tmp_path):
    report = _status(tmp_path)
    assert report["status"] == "ok"
    assert report["baseline_days"] == 14
    assert abs(report["latest_ratios"]["aggregator_buff163"] - 1.017) < 1e-9


def test_both_cny_venues_moving_alone_is_the_fx_signature(tmp_path):
    report = _status(tmp_path, {v: 1.06 for v in CNY_VENUES})
    assert report["status"] == "fx_signature"
    assert report["breached"] == sorted(CNY_VENUES)


def test_one_cny_venue_is_a_breach_not_a_signature(tmp_path):
    assert _status(tmp_path, {"aggregator_buff163": 0.95})["status"] == "cny_breach"


def test_csfloat_moving_is_not_an_fx_signature(tmp_path):
    # The denominator moving shifts every ratio, USD controls included.
    report = _status(tmp_path, {REFERENCE: 0.94})
    assert report["status"] == "cny_breach"
    assert all(abs(report["moves"][v]) > 0.03 for v in USD_CONTROLS)


def test_a_breach_warns_but_exits_zero(tmp_path, capsys, monkeypatch):
    _archive(tmp_path, {v: 1.06 for v in CNY_VENUES})
    monkeypatch.setattr("sys.argv", ["check_fx_ratios", "--archive-dir", str(tmp_path)])
    main()  # returns normally: warn-only
    out = capsys.readouterr().out.splitlines()
    assert json.loads(out[0])["status"] == "fx_signature"
    assert out[1].startswith("::warning::FX ratio monitor 2026-09-16: buff163 +6.0%")
