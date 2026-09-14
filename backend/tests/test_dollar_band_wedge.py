"""The two coverage predicates, and the attribution built on them.

The whole result rests on one distinction — a dollar miss that the REBASED band
covers is the wedge's fault and an anchor fix would recover it; a miss outside
both is the band's own. Getting that classification backwards would invert the
recommendation, so it is pinned here rather than trusted to a reading of the
table.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dollar_band_wedge.py"


@pytest.fixture(scope="module")
def wedge():
    spec = importlib.util.spec_from_file_location("dollar_band_wedge", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(**kw):
    base = dict(
        forecast_date=pd.Timestamp("2026-08-04").date(),
        horizon_days=3,
        base_price=100.0,
        actual_price=100.0,
        current_price=100.0,
        predicted_price_low=90.0,
        predicted_price_mid=100.0,
        predicted_price_high=110.0,
        model_version="lgbm-v3",
        base_stale_run_days=0.0,
    )
    base.update(kw)
    return base


def _prep(wedge, rows):
    return wedge._prepare(pd.DataFrame(rows))


def test_no_wedge_makes_the_two_predicates_identical(wedge):
    """The 40% of rows where the anchors agree must contribute nothing."""
    d = _prep(wedge, [_row(actual_price=105.0), _row(actual_price=130.0)])
    assert d["wedge"].tolist() == [0.0, 0.0]
    assert d["cov_dollar"].tolist() == d["cov_calibrated"].tolist() == [True, False]
    assert not d["recoverable"].any()


def test_a_miss_the_wedge_explains_is_recoverable(wedge):
    """Quoted off 50 while the outcome resolved off 100: the published band is
    [45, 55] but the calibrated one is [90, 110], and 105 lands in the latter."""
    d = _prep(
        wedge,
        [
            _row(
                current_price=50.0,
                actual_price=105.0,
                predicted_price_low=45.0,
                predicted_price_mid=50.0,
                predicted_price_high=55.0,
            )
        ],
    )
    assert not d["cov_dollar"].iloc[0]
    assert d["cov_calibrated"].iloc[0]
    assert d["recoverable"].iloc[0]
    assert not d["genuine_miss"].iloc[0]


def test_a_miss_outside_both_bands_is_genuine(wedge):
    """No anchor fix reaches this one, and it must not be counted as if it did."""
    d = _prep(
        wedge,
        [
            _row(
                current_price=50.0,
                actual_price=500.0,
                predicted_price_low=45.0,
                predicted_price_mid=50.0,
                predicted_price_high=55.0,
            )
        ],
    )
    assert not d["cov_dollar"].iloc[0]
    assert not d["cov_calibrated"].iloc[0]
    assert d["genuine_miss"].iloc[0]
    assert not d["recoverable"].iloc[0]


def test_every_dollar_miss_is_exactly_one_of_the_two(wedge):
    """The attribution must partition the misses, or the shares do not sum."""
    d = _prep(
        wedge,
        [
            _row(actual_price=105.0),  # covered
            _row(
                current_price=50.0, actual_price=105.0, predicted_price_low=45.0, predicted_price_high=55.0
            ),  # recoverable
            _row(
                current_price=50.0, actual_price=500.0, predicted_price_low=45.0, predicted_price_high=55.0
            ),  # genuine
        ],
    )
    misses = ~d["cov_dollar"]
    assert (d["recoverable"] ^ d["genuine_miss"])[misses].all()
    assert not (d["recoverable"] | d["genuine_miss"])[~misses].any()


def test_the_wedge_is_signed_base_over_quote(wedge):
    d = _prep(wedge, [_row(base_price=110.0, current_price=100.0)])
    assert d["wedge"].iloc[0] == pytest.approx(0.10)


def test_a_missing_quote_falls_back_to_base_and_reports_no_wedge(wedge):
    """Legacy rows predating current_price must not read as a 100% wedge."""
    for bad in (None, 0.0):
        d = _prep(wedge, [_row(current_price=bad)])
        assert d["wedge"].iloc[0] == pytest.approx(0.0)
