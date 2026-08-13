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
