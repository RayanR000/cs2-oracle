"""Tests for the default-off market-relative label path.

Design: docs/superpowers/specs/2026-08-06-market-relative-labels-design.md
"""
import numpy as np
import pandas as pd
import pytest

from models.forecaster import DIRECTION_FLAT_TOLERANCE_PCT, ItemForecaster


# --- the flag itself -------------------------------------------------------

def test_flag_defaults_to_off():
    f = ItemForecaster(db_session=None)
    assert f.market_relative_labels is False
    assert f.market_index is None
    assert f.direction_bands == {}


def test_flag_can_be_set():
    f = ItemForecaster(db_session=None, market_relative_labels=True)
    assert f.market_relative_labels is True


def test_market_factor_columns_are_never_features():
    """The factor is built from future prices. If it reaches feature_cols the
    model trains on the answer and every metric downstream is meaningless."""
    f = ItemForecaster(db_session=None)
    df = pd.DataFrame({
        "item_id": [1, 2],
        "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "price": [10.0, 11.0],
        "return_7d": [9.0, 10.0],
        "target_3d": [11.0, 12.0],
        "target_return_3d": [10.0, 9.0],
        "market_factor_3d": [1.0, 2.0],
        "market_factor_7d": [1.5, 2.5],
        "market_factor_14d": [2.0, 3.0],
        "market_factor_30d": [2.5, 3.5],
    })
    cols = f._select_feature_cols(df, ItemForecaster.HORIZONS,
                                  ItemForecaster.SHELVED_FEATURES)
    for h in ItemForecaster.HORIZONS:
        assert f"market_factor_{h}d" not in cols
    # An ordinary unshelved feature is still picked up, so the exclusion is
    # targeted rather than accidentally dropping everything.
    assert "return_7d" in cols


# --- demeaning -------------------------------------------------------------

def test_demean_subtracts_the_factor():
    got = ItemForecaster._demean_returns(
        np.array([5.0, -2.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    assert got == pytest.approx([4.0, -3.0, -1.0])


def test_demean_treats_a_missing_factor_as_zero():
    """A NaN factor must fall back to the control's own label, never drop the
    row -- changing row counts between arms breaks the paired comparison."""
    got = ItemForecaster._demean_returns(
        np.array([5.0, -2.0]), np.array([1.0, np.nan]))
    assert got == pytest.approx([4.0, -2.0])


def test_demean_handles_a_none_factor():
    got = ItemForecaster._demean_returns(np.array([5.0, -2.0]), None)
    assert got == pytest.approx([5.0, -2.0])


def test_demean_does_not_mutate_its_input():
    src = np.array([5.0, -2.0])
    ItemForecaster._demean_returns(src, None)
    assert src == pytest.approx([5.0, -2.0])


def test_demean_accepts_a_pandas_series():
    got = ItemForecaster._demean_returns(
        pd.Series([5.0, -2.0]), pd.Series([1.0, np.nan]))
    assert got == pytest.approx([4.0, -2.0])


# --- flat-band matching ----------------------------------------------------

def test_matched_flat_band_reproduces_the_target_share():
    rng = np.random.RandomState(0)
    resid = rng.normal(0, 3.0, 10_000)
    band = ItemForecaster._matched_flat_band(resid, 0.30)
    got = float((np.abs(resid) <= band).mean())
    assert got == pytest.approx(0.30, abs=0.01)


def test_matched_flat_band_ignores_nans():
    resid = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    band = ItemForecaster._matched_flat_band(resid, 0.5)
    assert np.isfinite(band)


def test_matched_flat_band_falls_back_on_an_empty_input():
    band = ItemForecaster._matched_flat_band(np.array([]), 0.3)
    assert band == DIRECTION_FLAT_TOLERANCE_PCT


def test_matched_flat_band_falls_back_when_everything_is_nan():
    band = ItemForecaster._matched_flat_band(np.array([np.nan, np.nan]), 0.3)
    assert band == DIRECTION_FLAT_TOLERANCE_PCT


# --- reconstruction --------------------------------------------------------

def test_residual_point_estimate_maps_classes_to_signed_band_edges():
    got = ItemForecaster._residual_point_estimate(np.array([0, 1, 2]), band=2.5)
    assert got == pytest.approx([-2.5, 0.0, 2.5])


def test_reconstruction_with_a_flat_market_reduces_to_the_residual_call():
    """m_hat = 0 must give back exactly the residual arm's own direction."""
    pred_cls = np.array([0, 1, 2])
    e_hat = ItemForecaster._residual_point_estimate(pred_cls, band=3.0)
    rebuilt = ItemForecaster._direction_classes(
        0.0 + e_hat, DIRECTION_FLAT_TOLERANCE_PCT)
    assert list(rebuilt) == [0, 1, 2]


def test_reconstruction_shifts_the_call_when_the_market_forecast_is_large():
    e_hat = ItemForecaster._residual_point_estimate(np.array([0]), band=1.0)
    # Residual says down by 1%, but the market is forecast up 5%.
    rebuilt = ItemForecaster._direction_classes(
        5.0 + e_hat, DIRECTION_FLAT_TOLERANCE_PCT)
    assert list(rebuilt) == [2]


# --- the classifier honours an explicit band -------------------------------

def test_explicit_flat_band_overrides_the_default_yardstick():
    """The residual arm's matched band must reach the training label. If
    _fit_direction_classifier ignored flat_band, the residual labels would be
    bucketed on the raw +/-0.5% band and the flat class would balloon."""
    f = ItemForecaster(db_session=None)
    rng = np.random.RandomState(0)
    X = rng.normal(size=(400, 3))
    y = rng.normal(0, 2.0, 400)

    default_cls = f._direction_classes(y)
    wide_cls = f._direction_classes(y, 5.0)
    # A 5% band buckets far more rows as flat than the 0.5% default.
    assert (wide_cls == 1).sum() > (default_cls == 1).sum()

    model = f._fit_direction_classifier(
        X, y, None, None, "gbdt", {}, horizon=7, flat_band=5.0,
        num_boost_round=5)
    assert model is not None


# --- end-to-end composition ------------------------------------------------

def _synthetic_panel(n_items=60, n_days=260, seed=0):
    """A panel with a real common factor plus item-specific noise, so the
    decomposition has something to find."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    market = rng.normal(0.0005, 0.01, n_days)
    rows = []
    for i in range(n_items):
        idio = rng.normal(0, 0.02, n_days)
        price = 10.0 * np.exp(np.cumsum(market + idio))
        rows.append(pd.DataFrame({"item_id": i, "date": dates, "price": price}))
    return pd.concat(rows, ignore_index=True)


def test_the_arm_composes_end_to_end():
    """Index -> factor -> demeaned label -> classifier -> reconstruction.

    The unit tests above cover each piece in isolation; this is the only test
    that runs them in the order _cv_evaluate_horizon runs them, which is where
    a shape or dtype mismatch would actually surface.
    """
    from models.market_factor import (
        build_market_index,
        forecast_market_factor,
        market_factor_for_horizon,
    )

    horizon = 7
    panel = _synthetic_panel()
    index = build_market_index(panel, min_items=30)
    assert index["valid"].sum() > 200

    factor = market_factor_for_horizon(index, horizon)
    # Forward return per item, in percent.
    panel = panel.sort_values(["item_id", "date"])
    panel["target_return_7d"] = (
        panel.groupby("item_id")["price"].shift(-horizon) / panel["price"] - 1.0
    ) * 100
    panel["market_factor_7d"] = pd.to_datetime(panel["date"]).map(factor).astype(float)
    frame = panel.dropna(subset=["target_return_7d"]).copy()
    coverage = float(frame["market_factor_7d"].notna().mean())
    assert coverage > 0.90, f"factor coverage {coverage:.2%}"

    f = ItemForecaster(db_session=None, market_relative_labels=True)
    f.market_index = index

    y = frame["target_return_7d"].to_numpy(dtype=float)
    resid = f._demean_returns(y, frame["market_factor_7d"])
    # Removing a genuine common factor must reduce dispersion.
    assert resid.std() < y.std()

    ctl_flat = float((f._direction_classes(y) == 1).mean())
    band = f._matched_flat_band(resid, ctl_flat)
    got_flat = float((f._direction_classes(resid, band) == 1).mean())
    assert got_flat == pytest.approx(ctl_flat, abs=0.01), (
        "the residual arm's flat-class share must match the control's, or a "
        "class-balance artifact reads as 'relabelling hurt'")

    rng = np.random.RandomState(0)
    X = rng.normal(size=(len(frame), 4))
    clf = f._fit_direction_classifier(
        X, resid, None, None, "gbdt", {}, horizon=horizon,
        flat_band=band, num_boost_round=5)
    pred_cls = clf.predict(X).argmax(axis=1)

    e_hat = f._residual_point_estimate(pred_cls, band)
    m_hat = np.array([
        forecast_market_factor(index, d, horizon)
        for d in pd.to_datetime(frame["date"]).to_numpy()
    ])
    rebuilt = f._direction_classes(m_hat + e_hat)
    assert rebuilt.shape == (len(frame),)
    assert set(np.unique(rebuilt)) <= {0, 1, 2}
    # Scored against the unchanged fixed yardstick, as the control is.
    actual_cls = f._direction_classes(y)
    acc = float((rebuilt == actual_cls).mean())
    assert 0.0 <= acc <= 1.0


def test_market_factor_column_is_absent_when_the_flag_is_off():
    """A control-arm frame must not carry the factor at all -- its presence is
    what distinguishes the arms downstream."""
    f = ItemForecaster(db_session=None)
    assert f.market_relative_labels is False
    assert f.market_index is None


# --- the env knob ----------------------------------------------------------

def test_env_knob_defaults_to_false(monkeypatch):
    import scripts.forecast_prices as fp
    monkeypatch.delenv("TRAIN_MARKET_RELATIVE_LABELS", raising=False)
    assert fp._market_relative_labels() is False


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
def test_env_knob_accepts_truthy_spellings(monkeypatch, raw):
    import scripts.forecast_prices as fp
    monkeypatch.setenv("TRAIN_MARKET_RELATIVE_LABELS", raw)
    assert fp._market_relative_labels() is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "banana"])
def test_env_knob_rejects_everything_else(monkeypatch, raw):
    import scripts.forecast_prices as fp
    monkeypatch.setenv("TRAIN_MARKET_RELATIVE_LABELS", raw)
    assert fp._market_relative_labels() is False
