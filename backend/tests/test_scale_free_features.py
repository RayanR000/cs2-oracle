"""Model features must be scale-free: a percentage target needs return-space inputs.

Measured on the shipped 2026-08-06 artifact, dollar-denominated columns carried
**55.6 / 70.2 / 77.5 / 86.6%** of total LightGBM gain at 3/7/14/30d — rising with
horizon, exactly like the served-cohort accuracy gap. `price_std_{7..60}d`,
`price_lag_*`, `price_log` and the MACD family are all in dollars, while
`target_return_{h}d` is a percentage.

A dollar-scale feature in a percent-return model is an item-identity proxy: the
tree splits on "is this a 5-cent item or a $50 item" and learns a per-tier base
rate. It cannot transfer across the served price range. The training median price
is $0.086 while served items reach $639, and `max_bin = 63` puts essentially
everything above ~$1 into one saturated terminal bin.

`models/conformal.py:48` already divides `price_std_60d` by `price` for precisely
this reason ("in dollars, so it is divided by price to make the scale return-space
and comparable across price tiers"). The band got the scale-free version; the
model did not.

The load-bearing test here is `test_model_features_are_invariant_to_price_scale`,
which asserts the *property* rather than a name list — a new dollar-scale feature
added later fails it without anyone remembering to update a list.
"""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from models.forecaster import ItemForecaster


@pytest.fixture
def forecaster(tmp_path_factory):
    return ItemForecaster(db_session=MagicMock(),
                          model_dir=str(tmp_path_factory.mktemp("saved_models")))


def _series(n_items=2, n_days=400, scale=1.0, start=date(2024, 1, 1)):
    """Dense daily series. `scale` multiplies every price, nothing else."""
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n_items):
        price = 10.0 + i
        for d in range(n_days):
            price *= 1.0 + rng.normal(0.0005, 0.02)
            rows.append({
                "item_id": f"item-{i}",
                "date": start + timedelta(days=d),
                "price": price * scale,
                "volume": 0.0,
            })
    return pd.DataFrame(rows)


_EVENTS = pd.DataFrame(columns=["date", "event_type", "name"])

# price_tier is deliberately exempt: it is a bounded categorical (0-4), not a
# continuous level a tree extrapolates from. It is the honest way to say "how
# expensive is this" without handing the model an unbounded dollar axis.
SCALE_DEPENDENT_BY_DESIGN = {"price_tier"}


def _model_features(forecaster, df):
    """The columns that actually reach a booster: select -> shelve -> allowlist."""
    cols = ItemForecaster._select_feature_cols(
        df, ItemForecaster.HORIZONS, ItemForecaster.SHELVED_FEATURES)
    return ItemForecaster._apply_feature_allowlist(
        cols, ItemForecaster.FEATURE_GROUP_ALLOWLIST)


class TestScaleInvariance:

    def test_model_features_are_invariant_to_price_scale(self, forecaster):
        """THE LOAD-BEARING TEST.

        Multiply every price by 100 and every served feature must be unchanged.
        A feature that moves is denominated in dollars and encodes item identity.
        """
        base = forecaster.engineer_features(_series(scale=1.0), _EVENTS)
        scaled = forecaster.engineer_features(_series(scale=100.0), _EVENTS)

        feats = [c for c in _model_features(forecaster, base)
                 if c not in SCALE_DEPENDENT_BY_DESIGN]
        assert feats, "precondition: there are model features to check"

        offenders = []
        for col in feats:
            a = base[col].to_numpy(dtype=float)
            b = scaled[col].to_numpy(dtype=float)
            both = np.isfinite(a) & np.isfinite(b)
            if not both.any():
                continue
            if not np.allclose(a[both], b[both], rtol=1e-6, atol=1e-9):
                denom = np.maximum(np.abs(a[both]), 1e-12)
                offenders.append(
                    f"{col} (max rel diff {np.max(np.abs(b[both] - a[both]) / denom):.3g})")

        assert not offenders, (
            "These model features change when every price is multiplied by 100, "
            "so they are dollar-denominated and act as item-identity proxies "
            "against a percentage target:\n  " + "\n  ".join(sorted(offenders))
        )

    def test_the_test_can_fail(self, forecaster):
        """Guard against the invariance test passing vacuously.

        A raw dollar column must be detected as scale-dependent by the same
        comparison the test above uses.
        """
        base = forecaster.engineer_features(_series(scale=1.0), _EVENTS)
        scaled = forecaster.engineer_features(_series(scale=100.0), _EVENTS)
        a = base["price_std_30d"].to_numpy(dtype=float)
        b = scaled["price_std_30d"].to_numpy(dtype=float)
        both = np.isfinite(a) & np.isfinite(b)
        assert both.any()
        assert not np.allclose(a[both], b[both], rtol=1e-6)


class TestCoefficientOfVariationColumns:

    @pytest.mark.parametrize("window", [7, 14, 20, 30, 60])
    def test_price_cv_is_the_std_over_price(self, forecaster, window):
        df = forecaster.engineer_features(_series(), _EVENTS)
        expected = df[f"price_std_{window}d"] / df["price"].replace(0, np.nan)
        got = df[f"price_cv_{window}d"]
        both = expected.notna() & got.notna()
        assert both.any()
        assert np.allclose(got[both], expected[both], rtol=1e-9)

    def test_cv_60d_matches_what_conformal_computes_for_sigma(self, forecaster):
        """conformal.sigma_from_columns is price_std_60d / price. The feature and
        the band must not drift apart into two definitions of the same thing."""
        from models import conformal
        df = forecaster.engineer_features(_series(), _EVENTS)
        # Non-binding floor/cap: this pins the *definition* (std/price), not the
        # clipping, which is a separate serving concern.
        sigma = conformal.sigma_from_columns(
            price_std_60d=df["price_std_60d"].to_numpy(dtype=float),
            price=df["price"].to_numpy(dtype=float),
            floor=1e-12, cap=1e12,
        )
        got = df["price_cv_60d"].to_numpy(dtype=float)
        both = np.isfinite(sigma) & np.isfinite(got)
        assert both.any()
        assert np.allclose(got[both], sigma[both], rtol=1e-9)


class TestDollarScaleColumnsAreShelved:
    """The raw dollar columns must still be COMPUTED (conformal, market
    aggregates and the log-return features all read them) but must not reach a
    booster."""

    DOLLAR_COLUMNS = [
        "price_std_7d", "price_std_14d", "price_std_20d",
        "price_std_30d", "price_std_60d",
        "price_log", "macd_line", "macd_signal", "macd_histogram",
    ]

    @pytest.mark.parametrize("col", DOLLAR_COLUMNS)
    def test_still_computed(self, forecaster, col):
        df = forecaster.engineer_features(_series(), _EVENTS)
        assert col in df.columns

    def test_no_dollar_column_survives_the_real_selection_and_prune(self, forecaster):
        """Mirrors test_no_volume_feature_survives_the_real_selection_and_prune.

        Shelving a feature removes what its partners correlated against, so a
        previously-pruned dollar column can survive the >0.95 prune and re-enter
        production. A name-list assertion cannot see that; running the real
        selection path can.
        """
        df = forecaster.engineer_features(_series(), _EVENTS)
        forecaster.feature_cols = ItemForecaster._select_feature_cols(
            df, ItemForecaster.HORIZONS, ItemForecaster.SHELVED_FEATURES)
        pruned = forecaster._prune_features(df)
        final = ItemForecaster._apply_feature_allowlist(
            pruned, ItemForecaster.FEATURE_GROUP_ALLOWLIST)

        leaked = [c for c in self.DOLLAR_COLUMNS if c in final]
        leaked += [c for c in final if c.startswith("price_lag_")]
        assert not leaked, f"dollar-scale columns reached the model: {leaked}"
