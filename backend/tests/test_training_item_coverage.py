"""The feature-row budget must reach the subsample, and stay decoupled.

train(max_rows=700_000) was passed at forecast_prices.py but never reached
max_feature_rows (default 100_000), so the model learned from 99 of the
5,377-item pool while serving forecasts for 8,691, and the caller's number was
a no-op that read as if it were doing something.

The fix is two named budgets, not one shared number:

  max_feature_rows  bounds the frame BEFORE feature engineering -> item coverage
  max_rows          caps each horizon's slice AFTER it

Unifying them would move both with one knob. It would also raise coverage
sharply: measured 2026-08-05, 700_000 feature rows (646 items) costs 468.7s
against 104.6s at 100_000 (99 items), exceeding the 462s the pre-rewrite
40-model grid cost. So the default is pinned to the measured status quo and
these tests guard the wiring, not a hoped-for improvement.
"""
from __future__ import annotations

import inspect

import pytest

from models.forecaster import ItemForecaster


def test_train_exposes_both_budgets_separately():
    sig = inspect.signature(ItemForecaster.train)
    assert "max_rows" in sig.parameters
    assert "max_feature_rows" in sig.parameters, (
        "the feature-row budget must be its own parameter; sharing max_rows "
        "makes one number move both item coverage and the per-horizon cap"
    )


def test_train_forwards_the_feature_budget():
    src = inspect.getsource(ItemForecaster.train)
    assert "max_feature_rows=max_feature_rows" in src, (
        "train must forward its feature budget to build_training_data, or the "
        "argument is silently ignored and the 100_000 default wins"
    )


def test_feature_budget_default_is_the_measured_status_quo():
    sig = inspect.signature(ItemForecaster.train)
    assert sig.parameters["max_feature_rows"].default == 100_000, (
        "raising this default changes training wall-clock ~linearly in rows "
        "and must be justified by a measured accuracy gain"
    )


def test_train_forwards_the_actual_budget_value(monkeypatch):
    """The forwarded value is the caller's, and it is not max_rows.

    Guards two near-misses at once: a literal wired in place of the argument,
    and the budgets being silently re-coupled.
    """
    seen = {}

    class _Abort(RuntimeError):
        pass

    def _capture(self, *args, **kwargs):
        seen.update(kwargs)
        raise _Abort

    monkeypatch.setattr(ItemForecaster, "build_training_data", _capture)

    forecaster = ItemForecaster.__new__(ItemForecaster)
    with pytest.raises(_Abort):
        ItemForecaster.train(forecaster, max_rows=700_000,
                             max_feature_rows=555_000)

    assert seen.get("max_feature_rows") == 555_000, (
        f"train forwarded max_feature_rows={seen.get('max_feature_rows')!r}, "
        "expected its own max_feature_rows argument"
    )


def test_the_per_horizon_cap_does_not_leak_into_coverage(monkeypatch):
    """max_rows must not reach the subsample.

    This is the regression that would silently 4.5x the daily retrain: the
    production caller passes max_rows=700_000, so if that value reaches
    build_training_data the budget jumps from 99 items to 646 with no code
    change visible at the call site.
    """
    seen = {}

    class _Abort(RuntimeError):
        pass

    def _capture(self, *args, **kwargs):
        seen.update(kwargs)
        raise _Abort

    monkeypatch.setattr(ItemForecaster, "build_training_data", _capture)

    forecaster = ItemForecaster.__new__(ItemForecaster)
    with pytest.raises(_Abort):
        ItemForecaster.train(forecaster, max_rows=700_000)

    assert seen.get("max_feature_rows") == 100_000, (
        f"max_rows leaked into coverage: subsample got "
        f"{seen.get('max_feature_rows')!r} when only max_rows was passed"
    )


class TestFeatureRowsOverride:
    """TRAIN_FEATURE_ROWS exists so a budget sweep needs no source edits.

    A silently-ignored or silently-zero budget is the same class of failure this
    module is about, so a bad value falls back loudly rather than shrinking the
    training set to nothing.
    """

    @staticmethod
    def _fn():
        from scripts.forecast_prices import _train_feature_rows
        return _train_feature_rows

    def test_default_when_unset(self, monkeypatch):
        from scripts.forecast_prices import DEFAULT_TRAIN_FEATURE_ROWS
        monkeypatch.delenv("TRAIN_FEATURE_ROWS", raising=False)
        assert self._fn()() == DEFAULT_TRAIN_FEATURE_ROWS

    def test_override_is_honoured(self, monkeypatch):
        monkeypatch.setenv("TRAIN_FEATURE_ROWS", "400000")
        assert self._fn()() == 400_000

    @pytest.mark.parametrize("bad", ["", "lots", "0", "-1"])
    def test_bad_values_fall_back_to_the_default(self, monkeypatch, bad):
        from scripts.forecast_prices import DEFAULT_TRAIN_FEATURE_ROWS
        monkeypatch.setenv("TRAIN_FEATURE_ROWS", bad)
        assert self._fn()() == DEFAULT_TRAIN_FEATURE_ROWS

    def test_production_default_matches_the_forecaster_default(self):
        """One status quo, stated in two places — they must agree."""
        from scripts.forecast_prices import DEFAULT_TRAIN_FEATURE_ROWS
        sig = inspect.signature(ItemForecaster.train)
        assert (DEFAULT_TRAIN_FEATURE_ROWS
                == sig.parameters["max_feature_rows"].default)
