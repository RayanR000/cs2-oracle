"""A NULL volume column must not crash feature engineering.

2026-08-08 is the first archive day whose `volume` is mostly NULL -- 33,613
non-null of 361,453 rows, against 100% populated on every earlier day. The
upstream feed stopped supplying it (`steam_volume.json` no longer exists;
see Track D2).

DuckDB hands a column containing NULLs to pandas as a nullable dtype, so
`(series > 0)` is BooleanDtype carrying pd.NA and `.astype(int)` raises
"cannot convert NA to integer". That aborted the first mode=full run in weeks,
in CI, and no test caught it because every fixture had a fully-populated
volume column.

The expected value is False -- "no volume confirmation" -- which is what the
numpy path already produced, since `NaN > 0` is False.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from models.forecaster import ItemForecaster


def _frame(volume) -> pd.DataFrame:
    n = len(volume)
    return pd.DataFrame(
        {
            "item_id": ["ak47"] * n,
            "date": pd.date_range("2026-01-01", periods=n, freq="D"),
            "price": np.linspace(10.0, 12.0, n),
            "volume": volume,
        }
    )


@pytest.fixture
def forecaster():
    return ItemForecaster(db_session=MagicMock())


def test_all_null_volume_does_not_crash(forecaster):
    """has_volume is False here, so the columns are filled with NaN wholesale."""
    df = _frame(pd.array([None] * 40, dtype="Int64"))
    out = forecaster._compute_price_features(df)
    assert out["volume_price_conf_7d"].isna().all()


def test_partially_null_volume_does_not_crash(forecaster):
    """The 2026-08-08 shape: mostly NULL, a minority populated. has_volume is
    True, so the log-change path runs and the astype(int) is reached."""
    vol = [100] * 5 + [None] * 35
    df = _frame(pd.array(vol, dtype="Int64"))
    out = forecaster._compute_price_features(df)  # must not raise
    assert "volume_price_conf_7d" in out
    assert "volume_price_conf_1d" in out


def test_null_rows_score_as_no_confirmation(forecaster):
    """An unknown volume change is not a confirmed one."""
    vol = [100] * 5 + [None] * 35
    df = _frame(pd.array(vol, dtype="Int64"))
    out = forecaster._compute_price_features(df)
    # Only where the return leg is itself defined: the first 7 rows have no
    # 7-day return, and NaN * 0 is NaN for reasons that have nothing to do
    # with volume.
    unknown = out["volume_log_change_7d"].isna() & out["return_7d"].notna()
    assert unknown.any(), "fixture did not produce any unknown rows"
    assert (out.loc[unknown, "volume_price_conf_7d"] == 0).all()


def test_nullable_and_numpy_dtypes_agree(forecaster):
    """The fillna(False) must be a no-op wherever the numpy path already
    worked, or it is a silent behaviour change on 13 years of data."""
    vol = [100, 120, 90, 140, 110] * 8
    nullable = forecaster._compute_price_features(_frame(pd.array(vol, dtype="Int64")))
    numpy_ = forecaster._compute_price_features(_frame(np.array(vol, dtype=float)))
    for col in ("volume_price_conf_7d", "volume_price_conf_1d"):
        pd.testing.assert_series_equal(nullable[col].astype(float), numpy_[col].astype(float), check_names=False)
