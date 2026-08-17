"""Supply-churn band-width features off the deep `buff_listing_count` sidecar.

The MAGNITUDE of a day-over-day BUFF listing change predicts forward |return|
(docs/research/2026-08-17-supply-churn-volatility-signal.md); the existing
`supply_change_7d` is signed and null for volatility. These features surface
`buff_listing_count` (joined today, consumed by nothing) as a churn magnitude.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.forecaster import ItemForecaster


def _df(*rows):
    return pd.DataFrame(
        [{"item_id": i, "date": pd.Timestamp(d).date(),
          "buff_listing_count": c} for i, d, c in rows]
    )


def test_absent_sidecar_is_safe():
    """No buff_listing_count column: churn is NaN, presence 0, no exception."""
    df = pd.DataFrame({"item_id": ["a"], "date": [pd.Timestamp("2023-01-01").date()]})
    out = ItemForecaster._compute_supply_churn_features(df)
    assert out["supply_churn_present"].iloc[0] == 0
    assert np.isnan(out["supply_churn"].iloc[0])
    assert np.isnan(out["supply_churn_abs"].iloc[0])


def test_churn_is_the_log_difference():
    out = ItemForecaster._compute_supply_churn_features(_df(
        ("a", "2023-01-01", 100.0),
        ("a", "2023-01-02", 200.0),
    )).sort_values("date")
    expected = np.log1p(200.0) - np.log1p(100.0)
    assert out["supply_churn"].iloc[1] == pytest.approx(expected)
    assert out["supply_churn_abs"].iloc[1] == pytest.approx(abs(expected))
    # first row has no prior -> NaN
    assert np.isnan(out["supply_churn"].iloc[0])


def test_magnitude_is_sign_free():
    """A drop and an equal-ratio rise give the same supply_churn_abs."""
    up = ItemForecaster._compute_supply_churn_features(_df(
        ("a", "2023-01-01", 100.0), ("a", "2023-01-02", 400.0))).sort_values("date")
    down = ItemForecaster._compute_supply_churn_features(_df(
        ("b", "2023-01-01", 400.0), ("b", "2023-01-02", 100.0))).sort_values("date")
    assert up["supply_churn_abs"].iloc[1] == pytest.approx(down["supply_churn_abs"].iloc[1])
    # signed churn has opposite signs
    assert up["supply_churn"].iloc[1] > 0 > down["supply_churn"].iloc[1]


def test_diff_never_crosses_item_boundary():
    """Two items interleaved by date: each item's first row is NaN, never a
    diff against the other item's last listing count."""
    out = ItemForecaster._compute_supply_churn_features(_df(
        ("a", "2023-01-01", 100.0),
        ("b", "2023-01-01", 5000.0),
        ("a", "2023-01-02", 110.0),
        ("b", "2023-01-02", 5100.0),
    ))
    for item in ("a", "b"):
        rows = out[out["item_id"] == item].sort_values("date")
        assert np.isnan(rows["supply_churn"].iloc[0])  # first obs, no prior
        assert not np.isnan(rows["supply_churn"].iloc[1])  # second obs, real diff


def test_zero_listing_days_are_not_churn():
    """A 0 listing count is 'no supply observed', not a real level: churn from or
    to it is NaN and presence is 0, so it cannot manufacture a huge log jump."""
    out = ItemForecaster._compute_supply_churn_features(_df(
        ("a", "2023-01-01", 0.0),
        ("a", "2023-01-02", 100.0),
    )).sort_values("date")
    assert out["supply_churn_present"].iloc[0] == 0
    assert np.isnan(out["supply_churn"].iloc[1])  # prior day was 0 -> NaN


def test_names_map_to_own_supply_churn_group():
    """Isolated group so an A/B admits only these, not the dormant supply_depth
    group's other columns."""
    from models.forecaster import _feature_group
    for n in ["supply_churn", "supply_churn_abs", "supply_churn_present"]:
        assert _feature_group(n) == "supply_churn"

