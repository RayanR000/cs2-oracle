# backend/tests/test_sidecar_attach.py
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.forecaster import ItemForecaster


def _daily():
    return pd.DataFrame(
        {
            "item_id": ["AK | X", "AK | X"],
            "date": [date(2026, 5, 1), date(2026, 5, 2)],
            "price": [10.0, 11.0],
            "volume": [0, 0],  # archive volume is dead
        }
    )


def test_volume_overwritten_from_sidecar(tmp_path, monkeypatch):
    vp = pd.DataFrame(
        {"item_id": ["AK | X"], "date": [date(2026, 5, 1)], "steam_volume": [55], "steam_sale_median": [10.0]}
    )
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


def test_bid_column_added_when_present(tmp_path, monkeypatch):
    monkeypatch.setenv("BID_FEATURES", "1")
    bp = pd.DataFrame({"item_id": ["AK | X"], "date": [date(2026, 5, 1)], "buff_bid": [8.5]})
    bp.to_parquet(tmp_path / "bid-panel.parquet", index=False)
    f = ItemForecaster.__new__(ItemForecaster)
    f.archive_dir = tmp_path
    out = f._attach_sidecars(_daily())
    assert out.loc[out["date"] == date(2026, 5, 1), "buff_bid"].iloc[0] == 8.5
    assert pd.isna(out.loc[out["date"] == date(2026, 5, 2), "buff_bid"].iloc[0])


def test_raw_sidecar_columns_excluded_from_feature_cols():
    """Raw sidecar literals must never be selectable, even with an empty shelf.

    Belt-and-braces: today they're kept out only because _feature_group() maps
    them to "other", which the default allowlist drops. The BID/STATTRAK A/B
    widens the allowlist to admit "other", which would then admit these RAW
    columns too -- and buff_listing_count has no gating flag at all, so it
    would leak training-only availability into the model. Exclusion must be
    structural (named in _select_feature_cols), not incidental to the shelf.
    """
    df = pd.DataFrame(
        {
            "item_id": ["AK | X"],
            "date": [date(2026, 5, 1)],
            "price": [10.0],
            "volume": [55.0],
            "buff_bid": [8.5],
            "st_premium": [1.2],
            "buff_listing_count": [40.0],
            "steam_sale_median": [9.9],
        }
    )
    selected = ItemForecaster._select_feature_cols(df, horizons=[3, 7, 14, 30], shelved=set())
    for raw_col in ("buff_bid", "st_premium", "buff_listing_count", "steam_sale_median"):
        assert raw_col not in selected
    # "volume" itself was already excluded pre-existing (raw literal, like
    # "price") -- this change did not touch that. Confirm price/volume
    # exclusion is unaffected by the new additions, not accidentally widened.
    assert "price" not in selected
    assert "volume" not in selected


def test_merge_normalizes_sidecar_date_dtype(tmp_path, monkeypatch):
    """A Timestamp-typed sidecar date must still join onto daily's python date.

    _attach_sidecars normalizes the sidecar's date column before merging, so a
    dtype drift between a sidecar writer and `daily` can't silently zero out
    coverage via an all-NaN merge.
    """
    monkeypatch.setenv("BID_FEATURES", "1")
    bp = pd.DataFrame(
        {
            "item_id": ["AK | X"],
            "date": pd.to_datetime([pd.Timestamp(2026, 5, 1)]),
            "buff_bid": [8.5],
        }
    )
    bp.to_parquet(tmp_path / "bid-panel.parquet", index=False)
    f = ItemForecaster.__new__(ItemForecaster)
    f.archive_dir = tmp_path
    out = f._attach_sidecars(_daily())
    matched = out.loc[out["date"] == date(2026, 5, 1), "buff_bid"].iloc[0]
    assert not pd.isna(matched)
    assert matched == 8.5


def test_flag_gated_sidecars_are_not_read_when_their_flag_is_off(tmp_path, monkeypatch):
    """bid / stattrak / supply-history feed only BID_FEATURES, STATTRAK_FEATURE
    and SUPPLY_CHURN_FEATURES. With each flag off its file is not read, and the
    frame keeps its rows; with it on, the column joins."""
    panels = {
        "bid-panel.parquet": ("buff_bid", "BID_FEATURES"),
        "stattrak-panel.parquet": ("st_premium", "STATTRAK_FEATURE"),
        "supply-history.parquet": ("buff_listing_count", "SUPPLY_CHURN_FEATURES"),
    }
    for fname, (col, _) in panels.items():
        pd.DataFrame({"item_id": ["AK | X"], "date": [date(2026, 5, 1)], col: [1.0]}).to_parquet(
            tmp_path / fname, index=False
        )
    f = ItemForecaster.__new__(ItemForecaster)
    f.archive_dir = tmp_path
    for _, flag in panels.values():
        monkeypatch.delenv(flag, raising=False)
    out = f._attach_sidecars(_daily(), include_volume_panel=False)
    assert list(out.columns) == ["item_id", "date", "price", "volume"]
    assert len(out) == 2
    for _, (col, flag) in panels.items():
        monkeypatch.setenv(flag, "1")
        assert col in f._attach_sidecars(_daily(), include_volume_panel=False).columns
        monkeypatch.delenv(flag)
