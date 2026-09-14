import io
import json
import sys
import zipfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.backfill_buff_iflow import (
    RESTRUCTURE,
    VOL_POST_RESTRUCTURE,
    VOL_PRE_RESTRUCTURE,
    parse_dump,
)

SLUG = "AK-47 | Redline (Field-Tested)"


def _zip_bytes(records) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("dump.json", "\n".join(json.dumps(r) for r in records))
    return buf.getvalue()


def _old_era_record(**kw):
    rec = {
        "appid": 730,
        "hash_name": SLUG,
        "buff_reference_price": 100.0,
        "count_in_24": 7,
        "buff_buy_num": 3,
        "buff_sell_num": 5,
    }
    rec.update(kw)
    return rec


def _new_era_record(**kw):
    rec = {
        "appid": 730,
        "hash_name": SLUG,
        "buff_sell": {"price": 100.0, "count": 5},
        "buff_buy": {"count": 3},
        "steam_volume": {"volume": 9},
    }
    rec.update(kw)
    return rec


def test_pre_restructure_row_populates_pre_column_only():
    out = parse_dump(_zip_bytes([_old_era_record()]), date(2024, 2, 1))
    assert len(out) == 1
    row = out[0]
    assert row[VOL_PRE_RESTRUCTURE] == 7
    assert row[VOL_POST_RESTRUCTURE] is None


def test_boundary_day_takes_old_branch():
    out = parse_dump(_zip_bytes([_old_era_record()]), RESTRUCTURE)
    assert len(out) == 1
    assert out[0][VOL_PRE_RESTRUCTURE] == 7
    assert out[0][VOL_POST_RESTRUCTURE] is None


def test_post_restructure_row_populates_post_column_only():
    out = parse_dump(_zip_bytes([_new_era_record()]), date(2024, 3, 1))
    assert len(out) == 1
    row = out[0]
    assert row[VOL_POST_RESTRUCTURE] == 9
    assert row[VOL_PRE_RESTRUCTURE] is None


def test_eras_never_share_a_row_and_no_merged_column():
    old = parse_dump(_zip_bytes([_old_era_record()]), date(2023, 5, 5))
    new = parse_dump(_zip_bytes([_new_era_record()]), date(2025, 5, 5))
    for row in old + new:
        assert "steam_vol" not in row
        assert "steam_volume" not in row
        populated = [k for k in (VOL_PRE_RESTRUCTURE, VOL_POST_RESTRUCTURE) if row[k] is not None]
        assert len(populated) == 1, f"era splice: {populated}"


def test_non_cs_missing_slug_and_nonpositive_price_dropped():
    recs = [
        _old_era_record(appid=570),  # Dota, not CS
        _old_era_record(hash_name=None),
        _old_era_record(buff_reference_price=0),
        _new_era_record(buff_sell={"price": -1.0, "count": 1}),
    ]
    assert parse_dump(_zip_bytes(recs), date(2024, 2, 1)) == []
