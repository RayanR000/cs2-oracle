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


class TestMedianPriceFloor:
    """The universe filter that decides *which* items the budget may buy.

    Training applied no price filter, so at the 100K default only ~20 of the
    ~99 selected items were in the >= $1 cohort production serves, and 44% of
    the pool is stickers and graffiti at a $0.03 median. The floor exists so a
    budget above ~1.0M can cover the 926-item served cohort outright — the only
    configuration that removes the subsample's item-draw variance (measured
    sd 1.5-3.1pp on mean_classifier_acc_ge1 across 8 seeds) rather than
    shrinking it.
    """

    @staticmethod
    def _frame():
        import pandas as pd
        return pd.DataFrame({
            "item_id": ["cheap"] * 3 + ["dear"] * 3 + ["spiky"] * 3,
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"] * 3),
            # spiky is a penny item with one large print: its mean clears $1
            # but its median does not.
            "price": [0.03, 0.04, 0.05,
                      5.00, 6.00, 7.00,
                      0.03, 0.04, 99.0],
        })

    def test_keeps_only_items_whose_median_clears_the_floor(self):
        out = ItemForecaster._filter_by_median_price(self._frame(), 1.0)
        assert set(out["item_id"]) == {"dear"}

    def test_uses_the_median_not_the_mean(self):
        """A single spike must not promote a penny item into the universe."""
        frame = self._frame()
        spiky = frame[frame["item_id"] == "spiky"]
        assert spiky["price"].mean() > 1.0, "fixture no longer tests the case"
        out = ItemForecaster._filter_by_median_price(frame, 1.0)
        assert "spiky" not in set(out["item_id"])

    def test_whole_item_histories_survive(self):
        """Row-level filtering would corrupt lag/rolling features."""
        out = ItemForecaster._filter_by_median_price(self._frame(), 1.0)
        assert len(out) == 3, "the kept item must keep every one of its rows"

    def test_is_a_no_op_when_every_item_clears(self):
        frame = self._frame()
        out = ItemForecaster._filter_by_median_price(frame, 0.001)
        assert len(out) == len(frame)

    def test_build_training_data_exposes_and_defaults_to_no_filter(self):
        sig = inspect.signature(ItemForecaster.build_training_data)
        assert "min_median_price" in sig.parameters
        assert sig.parameters["min_median_price"].default is None, (
            "the default universe must stay the pool, or this lands as a "
            "silent production change"
        )

    def test_train_exposes_and_forwards_the_floor(self):
        sig = inspect.signature(ItemForecaster.train)
        assert "min_median_price" in sig.parameters
        assert sig.parameters["min_median_price"].default is None
        assert "min_median_price=min_median_price" in inspect.getsource(
            ItemForecaster.train)

    def test_floor_is_applied_before_the_subsample(self):
        """Order is the whole point: filtering after the subsample would spend
        the row budget on the pool and then throw most of it away."""
        src = inspect.getsource(ItemForecaster.build_training_data)
        assert src.index("_filter_by_median_price") < src.index(
            "_stratified_item_subsample"), (
            "the price floor must narrow the universe BEFORE the budget is "
            "spent, otherwise it cannot buy served-cohort breadth"
        )


class TestTrainMinMedianPriceEnv:
    @staticmethod
    def _fn():
        from scripts.forecast_prices import _train_min_median_price
        return _train_min_median_price

    def test_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("TRAIN_MIN_MEDIAN_PRICE", raising=False)
        assert self._fn()() is None

    def test_override_is_honoured(self, monkeypatch):
        monkeypatch.setenv("TRAIN_MIN_MEDIAN_PRICE", "1.0")
        assert self._fn()() == 1.0

    @pytest.mark.parametrize("bad", ["", "dollars", "0", "-1"])
    def test_bad_values_disable_the_floor(self, monkeypatch, bad):
        monkeypatch.setenv("TRAIN_MIN_MEDIAN_PRICE", bad)
        assert self._fn()() is None
