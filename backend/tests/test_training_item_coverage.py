"""The feature-row budget must reach the subsample, and stay decoupled.

train(max_rows=700_000) was passed at forecast_prices.py but never reached
max_feature_rows (default 100_000), so the model learned from 99 of the
5,377-item pool while serving forecasts for 8,691, and the caller's number was
a no-op that read as if it were doing something.

The fix is two named budgets, not one shared number:

  max_feature_rows  bounds the frame BEFORE feature engineering -> item coverage
  max_rows          caps each horizon's slice AFTER it

Unifying them would move both with one knob, which is why they stay separate
even now that they hold the same value.

Both defaults moved on 2026-08-08 (step 7): the budget to 1_200_000 and the
median-price floor to $1, together, because they are one setting. At the floor
the served cohort is 926 items / 993,464 item-days, so that budget covers it
with no subsample — and the subsample is what these tests were originally
written around. The justification is measurability, not accuracy: the draw's
own seed moves `mean_classifier_acc_ge1` by sd 1.5-3.1pp, and the +3.50pp that
once also justified the floor does not reproduce
(`docs/changelog/2026-08-08-per-fold-price-filter-rederived.md`).
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


def test_feature_budget_default_covers_the_served_cohort_outright():
    """Shipped 2026-08-08: the budget must clear the >= $1 cohort's row count.

    Paired with the $1 floor, 1.2M covers the 926-item / 993,464-row served
    cohort with no subsample at all. Dropping below ~1.0M re-engages
    `_stratified_item_subsample` over the *filtered* universe, which is a
    smaller draw rather than no draw — the failure mode the floor exists to
    remove. Raise the floor and this number has to be re-measured with it.
    """
    sig = inspect.signature(ItemForecaster.train)
    assert sig.parameters["max_feature_rows"].default == 1_200_000, (
        "the feature budget and the median-price floor are one setting; "
        "changing either alone leaves an item draw in the training set"
    )


def test_price_floor_default_matches_the_served_cohort():
    """`train()` bare must be production's configuration, floor included.

    A bare `train()` that raised the budget but not the floor would spend 12x
    the wall-clock buying the pool's tier mix — 44% stickers and graffiti at a
    $0.03 median — which is the opposite of what the budget was raised for.
    """
    sig = inspect.signature(ItemForecaster.train)
    assert sig.parameters["min_median_price"].default == 1.0, (
        "the budget default assumes the floor; unset, it buys pooled breadth"
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
        ItemForecaster.train(forecaster, max_rows=700_000, max_feature_rows=555_000)

    assert seen.get("max_feature_rows") == 555_000, (
        f"train forwarded max_feature_rows={seen.get('max_feature_rows')!r}, expected its own max_feature_rows argument"
    )


def test_the_per_horizon_cap_does_not_leak_into_coverage(monkeypatch):
    """max_rows must not reach the subsample.

    The two budgets now sit at the same value in production, so this test
    passes a deliberately different max_rows: with both at 1.2M a leak would
    be invisible, and the separation is the property under test, not the
    numbers.
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

    assert seen.get("max_feature_rows") == 1_200_000, (
        f"max_rows leaked into coverage: subsample got {seen.get('max_feature_rows')!r} when only max_rows was passed"
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
        assert sig.parameters["max_feature_rows"].default == DEFAULT_TRAIN_FEATURE_ROWS


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

        return pd.DataFrame(
            {
                "item_id": ["cheap"] * 3 + ["dear"] * 3 + ["spiky"] * 3,
                "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"] * 3),
                # spiky is a penny item with one large print: its mean clears $1
                # but its median does not.
                "price": [0.03, 0.04, 0.05, 5.00, 6.00, 7.00, 0.03, 0.04, 99.0],
            }
        )

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
        """The lower-level builder keeps the pool; only `train()` ships a floor.

        Every direct caller of `build_training_data` is a test or a research
        harness choosing its own universe deliberately. Defaulting a floor here
        would change what those arms measure without touching their code —
        production reaches it through `train()`, which does default the floor.
        """
        sig = inspect.signature(ItemForecaster.build_training_data)
        assert "min_median_price" in sig.parameters
        assert sig.parameters["min_median_price"].default is None

    def test_train_exposes_and_forwards_the_floor(self):
        sig = inspect.signature(ItemForecaster.train)
        assert "min_median_price" in sig.parameters
        assert sig.parameters["min_median_price"].default == 1.0
        assert "min_median_price=min_median_price" in inspect.getsource(ItemForecaster.train)

    def test_floor_is_applied_before_the_subsample(self):
        """Order is the whole point: filtering after the subsample would spend
        the row budget on the pool and then throw most of it away."""
        src = inspect.getsource(ItemForecaster.build_training_data)
        assert src.index("_filter_by_median_price") < src.index("_stratified_item_subsample"), (
            "the price floor must narrow the universe BEFORE the budget is "
            "spent, otherwise it cannot buy served-cohort breadth"
        )


class TestFoldMedianPriceItems:
    """The look-ahead-free counterpart, used to re-derive the floor's +3.50pp.

    `_filter_by_median_price` selects on a median over the whole 2013-2026
    frame, so a 2019 fold trains on "items that were >= $1 at some point
    through 2026". The effect it was credited with is h=30 only and null at
    3/7/14, which is the signature of survivorship rather than of liquidity —
    so the number cannot adjudicate itself and this helper exists to re-derive
    it. See docs/superpowers/plans/2026-08-08-per-fold-price-filter.md.
    """

    @staticmethod
    def _frame():
        import pandas as pd

        # `riser` is the survivorship case: pennies before the cutoff, $50
        # after. `faller` is its mirror. Only `faller` was nameable in 2020.
        days = pd.to_datetime(["2020-01-01", "2020-01-02", "2026-01-01", "2026-01-02"])
        return pd.DataFrame(
            {
                "item_id": ["riser"] * 4 + ["faller"] * 4 + ["steady"] * 4,
                "date": list(days) * 3,
                "price": [0.03, 0.04, 50.0, 52.0, 50.0, 52.0, 0.03, 0.04, 5.0, 5.0, 5.0, 5.0],
            }
        )

    def test_an_item_that_only_clears_the_floor_later_is_excluded(self):
        import pandas as pd

        out = ItemForecaster._fold_median_price_items(self._frame(), 1.0, pd.Timestamp("2021-01-01"))
        assert "riser" not in out, "selecting on post-cutoff prices is the look-ahead this helper exists to remove"

    def test_an_item_that_cleared_it_and_collapsed_is_included(self):
        import pandas as pd

        out = ItemForecaster._fold_median_price_items(self._frame(), 1.0, pd.Timestamp("2021-01-01"))
        assert "faller" in out, (
            "the universe is what was knowable at the cutoff, not what survived to the end of the sample"
        )

    def test_the_full_sample_filter_disagrees_on_the_same_frame(self):
        """The two must differ here, or the fixture is not testing the leak."""
        import pandas as pd

        full = set(ItemForecaster._filter_by_median_price(self._frame(), 1.0)["item_id"])
        fold = ItemForecaster._fold_median_price_items(self._frame(), 1.0, pd.Timestamp("2021-01-01"))
        assert "riser" in full and "riser" not in fold

    def test_an_empty_pre_cutoff_window_yields_nothing(self):
        """Not everything: a fallback to "keep all" would silently restore the
        full-sample universe on the earliest folds, where the leak is largest."""
        import pandas as pd

        out = ItemForecaster._fold_median_price_items(self._frame(), 1.0, pd.Timestamp("2013-01-01"))
        assert out == set()

    def test_rows_on_the_cutoff_day_are_excluded(self):
        """`date < cutoff`, strictly. The cutoff is already the embargoed
        boundary, so admitting its own day reads one day of the purge."""
        import pandas as pd

        frame = self._frame()
        out = ItemForecaster._fold_median_price_items(frame, 1.0, pd.Timestamp("2020-01-01"))
        assert out == set(), (
            "only 2020-01-01 rows exist before this cutoff if the comparison "
            "is <=, and faller would clear the floor on them"
        )


class TestTrainMinMedianPriceEnv:
    @staticmethod
    def _fn():
        from scripts.forecast_prices import _train_min_median_price

        return _train_min_median_price

    def test_default_when_unset(self, monkeypatch):
        from scripts.forecast_prices import DEFAULT_TRAIN_MIN_MEDIAN_PRICE

        monkeypatch.delenv("TRAIN_MIN_MEDIAN_PRICE", raising=False)
        assert self._fn()() == DEFAULT_TRAIN_MIN_MEDIAN_PRICE

    def test_override_is_honoured(self, monkeypatch):
        monkeypatch.setenv("TRAIN_MIN_MEDIAN_PRICE", "5.0")
        assert self._fn()() == 5.0

    @pytest.mark.parametrize("bad", ["", "dollars"])
    def test_unparseable_values_keep_the_default_floor(self, monkeypatch, bad):
        """A typo must not silently widen the training universe.

        This flipped when the floor shipped: while the default was None a bad
        value could only fall back to "no filter", so it was indistinguishable
        from the default. Now it is the difference between the served cohort
        and the pool, and falling back to the pool is the dangerous direction.
        """
        from scripts.forecast_prices import DEFAULT_TRAIN_MIN_MEDIAN_PRICE

        monkeypatch.setenv("TRAIN_MIN_MEDIAN_PRICE", bad)
        assert self._fn()() == DEFAULT_TRAIN_MIN_MEDIAN_PRICE

    @pytest.mark.parametrize("off", ["0", "-1"])
    def test_non_positive_is_the_escape_hatch_to_the_pooled_universe(self, monkeypatch, off):
        """The one way back to the pre-2026-08-08 universe, and it is explicit."""
        monkeypatch.setenv("TRAIN_MIN_MEDIAN_PRICE", off)
        assert self._fn()() is None

    def test_the_production_default_matches_the_served_floor(self):
        """The training floor tracks what the product actually serves."""
        from api.serving_policy import MIN_SERVED_PRICE_USD
        from scripts.forecast_prices import DEFAULT_TRAIN_MIN_MEDIAN_PRICE

        assert DEFAULT_TRAIN_MIN_MEDIAN_PRICE == MIN_SERVED_PRICE_USD, (
            "training on a different cohort than the one served is the train/serve gap this knob closed"
        )
