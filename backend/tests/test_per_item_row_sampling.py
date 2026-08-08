"""Per-item row sampling: spend the row budget on item diversity, not depth.

`_build_production_split` caps an oversized training slice with
`train_set.sample(n=max_rows)` — a uniform draw across all rows, so an item's
share of the sample is its share of the rows. Under a price floor that is the
wrong axis. Measured 2026-08-07 on the paired harness (25-26 folds, 150 held-out
>= $1 items, `cluster_key="fold_id"`):

    ge1_budgeted    71K rows/fold   +5.72pp at 30d [+2.91, +9.02]
    ge1_full       728K rows/fold   +3.50pp at 30d [+1.56, +5.98]

Ten times the rows lost at both horizons. What pays is item diversity per row,
which replicates the breadth harness's `wide_unbudgeted` finding. The better arm
could not be expressed in production because every sampler in the training path
selects whole item histories: a 110K budget under the floor buys ~93 items at
full depth, never 728 items at ~98 rows each.

These tests cover the sampler that closes that gap. It mirrors
`scripts/ab_test_training_breadth.py::_stratified_sample`, which is the code the
+5.72pp was measured on — uniform-at-random within each item rather than the
most recent rows, so the full calendar window survives, which a `tail()` cap
destroyed once already (`2026-07-16-training-window-audit.md`).

Default off. Production is byte-identical until TRAIN_PER_ITEM_ROWS is set.
"""
from __future__ import annotations

import inspect

import pandas as pd
import pytest

from models.forecaster import ItemForecaster


def _frame(rows_per_item: dict, start: str = "2024-01-01") -> pd.DataFrame:
    """One row per item-day, `rows_per_item[item]` consecutive days each."""
    recs = []
    for item_id, n in rows_per_item.items():
        dates = pd.date_range(start, periods=n, freq="D").date
        recs.extend({"item_id": item_id, "date": d, "price": 1.0 + i}
                    for i, d in enumerate(dates))
    return pd.DataFrame(recs)


class TestQuota:
    """Every item gets the same quota, bounded by what it actually has."""

    def test_equalises_item_shares_on_a_skewed_frame(self):
        """The whole point: one deep item must not eat the budget.

        Uniform sampling gives the 1000-row item ~90% of a 110-row draw. A
        per-item quota gives it 10, the same as every other item.
        """
        df = _frame({"deep": 1000, **{f"shallow_{i}": 10 for i in range(10)}})
        out = ItemForecaster._per_item_row_sample(df, max_rows=110, seed=42)

        counts = out.groupby("item_id").size()
        assert counts.max() == counts.min() == 10, (
            f"quota is not equal across items: {counts.to_dict()}"
        )

    def test_keeps_every_item(self):
        """Diversity is the objective, so no item may be dropped."""
        df = _frame({f"item_{i}": 50 for i in range(20)})
        out = ItemForecaster._per_item_row_sample(df, max_rows=100, seed=42)

        assert set(out["item_id"]) == set(df["item_id"]), (
            "an item was dropped; a per-item sampler must thin items, not "
            "discard them — discarding is what the whole-history sampler "
            "already does"
        )

    def test_items_shorter_than_the_quota_keep_everything(self):
        df = _frame({"deep": 500, "short": 3})
        out = ItemForecaster._per_item_row_sample(df, max_rows=200, seed=42)

        assert (out["item_id"] == "short").sum() == 3
        assert (out["item_id"] == "deep").sum() == 100

    def test_never_exceeds_the_budget(self):
        """max_rows is a memory guard upstream, so the floor of one row per
        item must not be allowed to breach it when items outnumber the budget.
        """
        df = _frame({f"item_{i}": 5 for i in range(50)})
        out = ItemForecaster._per_item_row_sample(df, max_rows=20, seed=42)

        assert len(out) <= 20, f"returned {len(out)} rows against a 20 budget"

    def test_undersized_frames_pass_through_unchanged(self):
        df = _frame({"a": 10, "b": 10})
        out = ItemForecaster._per_item_row_sample(df, max_rows=1000, seed=42)

        pd.testing.assert_frame_equal(out, df)


class TestSamplingProperties:
    def test_preserves_the_calendar_window(self):
        """A tail() cap keeps recent rows only, which silently disabled
        expanding-window CV once. Uniform-within-item must not: with 40 items
        drawn independently, both ends of the window survive.
        """
        df = _frame({f"item_{i}": 400 for i in range(40)})
        out = ItemForecaster._per_item_row_sample(df, max_rows=4000, seed=42)

        span_in = (max(df["date"]) - min(df["date"])).days
        span_out = (max(out["date"]) - min(out["date"])).days
        assert span_out >= span_in * 0.95, (
            f"calendar window collapsed from {span_in}d to {span_out}d"
        )

    def test_is_deterministic_for_a_seed(self):
        df = _frame({f"item_{i}": 100 for i in range(10)})
        a = ItemForecaster._per_item_row_sample(df, max_rows=200, seed=42)
        b = ItemForecaster._per_item_row_sample(df, max_rows=200, seed=42)

        pd.testing.assert_frame_equal(a, b)

    def test_a_different_seed_draws_different_rows(self):
        df = _frame({f"item_{i}": 100 for i in range(10)})
        a = ItemForecaster._per_item_row_sample(df, max_rows=200, seed=42)
        b = ItemForecaster._per_item_row_sample(df, max_rows=200, seed=7)

        assert not a.index.equals(b.index), "the seed does not reach the draw"

    def test_returns_date_sorted_rows(self):
        """`_build_production_split` returns a date-sorted train_set and
        downstream code reads it that way.
        """
        df = _frame({f"item_{i}": 100 for i in range(10)})
        out = ItemForecaster._per_item_row_sample(df, max_rows=200, seed=42)

        assert out["date"].is_monotonic_increasing


class TestSplitWiring:
    """The sampler has to replace the uniform cap, and only when asked."""

    @staticmethod
    def _tdf(n_items: int = 30, n_days: int = 120) -> pd.DataFrame:
        df = _frame({f"item_{i}": n_days for i in range(n_items)})
        df["target_return_7d"] = 1.0
        return df.sort_values("date").reset_index(drop=True)

    def test_split_exposes_the_flag_defaulted_off(self):
        sig = inspect.signature(ItemForecaster._build_production_split)
        assert "per_item_row_sampling" in sig.parameters
        assert sig.parameters["per_item_row_sampling"].default is False, (
            "defaulting this on would change production's training set with "
            "no measurement behind it"
        )

    def test_off_leaves_the_uniform_cap_in_place(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            ItemForecaster, "_per_item_row_sample",
            staticmethod(lambda *a, **k: called.append(1)))

        forecaster = ItemForecaster.__new__(ItemForecaster)
        train_set, _ = ItemForecaster._build_production_split(
            forecaster, self._tdf(), horizon=7, max_rows=500)

        assert not called, "the per-item sampler ran with the flag off"
        assert len(train_set) <= 500

    def test_on_routes_the_cap_through_the_per_item_sampler(self, monkeypatch):
        seen = {}
        real = ItemForecaster._per_item_row_sample

        def _spy(train_set, max_rows, seed=42):
            seen["max_rows"] = max_rows
            seen["n_rows_in"] = len(train_set)
            return real(train_set, max_rows, seed)

        monkeypatch.setattr(ItemForecaster, "_per_item_row_sample",
                            staticmethod(_spy))

        forecaster = ItemForecaster.__new__(ItemForecaster)
        train_set, _ = ItemForecaster._build_production_split(
            forecaster, self._tdf(), horizon=7, max_rows=500,
            per_item_row_sampling=True)

        assert seen.get("max_rows") == 500, (
            f"the sampler got max_rows={seen.get('max_rows')!r}; the budget "
            "must be the caller's per-horizon cap"
        )
        assert len(train_set) <= 500

    def test_on_equalises_item_shares_in_the_train_set(self):
        """End-to-end through the split, on a frame skewed like the real one."""
        df = _frame({"deep": 400, **{f"shallow_{i}": 40 for i in range(20)}})
        df["target_return_7d"] = 1.0
        tdf = df.sort_values("date").reset_index(drop=True)

        forecaster = ItemForecaster.__new__(ItemForecaster)
        train_set, _ = ItemForecaster._build_production_split(
            forecaster, tdf, horizon=7, max_rows=210,
            per_item_row_sampling=True)

        counts = train_set.groupby("item_id").size()
        assert counts.max() - counts.min() <= 1, (
            f"item shares are still skewed through the split: "
            f"max={counts.max()} min={counts.min()}"
        )

    def test_the_validation_set_is_never_thinned(self):
        """Thinning val would change the evaluation cohort, which is the
        artifact the paired harness exists to avoid.
        """
        tdf = self._tdf()
        forecaster = ItemForecaster.__new__(ItemForecaster)

        _, val_off = ItemForecaster._build_production_split(
            forecaster, tdf, horizon=7, max_rows=100)
        _, val_on = ItemForecaster._build_production_split(
            forecaster, tdf, horizon=7, max_rows=100,
            per_item_row_sampling=True)

        pd.testing.assert_frame_equal(val_off, val_on)


class TestTrainWiring:
    """train -> _train_horizon_inline -> _build_production_split."""

    def test_train_exposes_the_flag_defaulted_off(self):
        sig = inspect.signature(ItemForecaster.train)
        assert "per_item_row_sampling" in sig.parameters
        assert sig.parameters["per_item_row_sampling"].default is False

    def test_horizon_inline_exposes_the_flag_defaulted_off(self):
        sig = inspect.signature(ItemForecaster._train_horizon_inline)
        assert "per_item_row_sampling" in sig.parameters
        assert sig.parameters["per_item_row_sampling"].default is False

    def test_train_forwards_the_flag_to_the_horizon_loop(self):
        src = inspect.getsource(ItemForecaster.train)
        assert "per_item_row_sampling=per_item_row_sampling" in src, (
            "train must forward the flag or the knob is a no-op"
        )

    def test_horizon_inline_forwards_the_flag_to_the_split(self):
        src = inspect.getsource(ItemForecaster._train_horizon_inline)
        assert "per_item_row_sampling=per_item_row_sampling" in src


class TestTrainPerItemRowsEnv:
    """TRAIN_PER_ITEM_ROWS, matching TRAIN_MIN_MEDIAN_PRICE's shape.

    Env-configured because forecast_prices.py parses argv as a plain set, so a
    two-token flag does not fit.
    """

    @staticmethod
    def _fn():
        from scripts.forecast_prices import _train_per_item_rows
        return _train_per_item_rows

    def test_unset_is_off(self, monkeypatch):
        monkeypatch.delenv("TRAIN_PER_ITEM_ROWS", raising=False)
        assert self._fn()() is False

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_values_enable_it(self, monkeypatch, raw):
        monkeypatch.setenv("TRAIN_PER_ITEM_ROWS", raw)
        assert self._fn()() is True

    @pytest.mark.parametrize("raw", ["0", "false", "no", "off", ""])
    def test_falsy_values_leave_it_off(self, monkeypatch, raw):
        monkeypatch.setenv("TRAIN_PER_ITEM_ROWS", raw)
        assert self._fn()() is False

    def test_an_unparseable_value_falls_back_off_not_on(self, monkeypatch):
        """A typo must not silently change the training set."""
        monkeypatch.setenv("TRAIN_PER_ITEM_ROWS", "banana")
        assert self._fn()() is False

    def test_the_production_caller_passes_it(self):
        from scripts import forecast_prices
        src = inspect.getsource(forecast_prices)
        assert "per_item_row_sampling=_train_per_item_rows()" in src, (
            "the knob must reach forecaster.train() from the production path"
        )
