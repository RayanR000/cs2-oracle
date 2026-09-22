"""What price serving quotes from, and the arm that narrows it.

`predict()` converts a return-space forecast to dollars against
`_smoothed_anchor_prices` -- the median of the last SMOOTH_WINDOW observations
within MAX_WINDOW_SPAN_DAYS. It detects items whose latest quote deviates >10%
from that median, logs `"using smoothed price"`, and then substitutes the
smoothed value for **every** item, outlier or not. So the served
`current_price` is a number no venue quoted, and the wedge `p[d]/S[d]` rides on
every served item rather than the spiky ones.

That wedge is what contaminates both label bases in opposite directions
(`docs/changelog/2026-08-11-smoothed-anchor-label-measured.md`), which is why
the arm attacks it at source in the serving basis instead of in the label.
Arm A of `docs/superpowers/plans/2026-08-11-serving-anchor-freshness.md`: make
the substitution conditional on the outlier test the comment already claims.
"""

from __future__ import annotations

import pandas as pd
import pytest
from models.forecaster import ItemForecaster

from tests._source import method_closure_source


def _pair(price, smoothed):
    """Aligned (latest quote, smoothed anchor) columns, as `predict` holds them."""
    frame = pd.DataFrame({"price": price, "_smoothed_price": smoothed})
    return frame["price"], frame["_smoothed_price"]


class TestTheFlag:
    def test_the_arm_is_off_by_default(self, monkeypatch):
        monkeypatch.delenv("SERVE_OUTLIER_GATED_ANCHOR", raising=False)
        assert ItemForecaster.outlier_gated_anchor_enabled() is False

    def test_only_an_explicit_1_enables_it(self, monkeypatch):
        """Matching every other instrument flag. `true` reading as on would put
        an arm into a production forecast run through a typo."""
        monkeypatch.setenv("SERVE_OUTLIER_GATED_ANCHOR", "true")
        assert ItemForecaster.outlier_gated_anchor_enabled() is False
        monkeypatch.setenv("SERVE_OUTLIER_GATED_ANCHOR", "1")
        assert ItemForecaster.outlier_gated_anchor_enabled() is True


class TestTheShippedDefault:
    def test_a_quote_that_agrees_with_its_median_is_still_replaced(self, monkeypatch):
        """The behaviour the plan found and the arm changes: the substitution is
        unconditional, so an item 2% off its median -- not an outlier by the
        code's own test -- is served the median anyway."""
        monkeypatch.delenv("SERVE_OUTLIER_GATED_ANCHOR", raising=False)
        price, smoothed = _pair([10.2], [10.0])
        base, deviates = ItemForecaster._serving_base_price(price, smoothed)
        assert base.iloc[0] == pytest.approx(10.0)
        assert not deviates.iloc[0]

    def test_a_deviating_quote_is_replaced(self, monkeypatch):
        monkeypatch.delenv("SERVE_OUTLIER_GATED_ANCHOR", raising=False)
        price, smoothed = _pair([15.0], [10.0])
        base, deviates = ItemForecaster._serving_base_price(price, smoothed)
        assert base.iloc[0] == pytest.approx(10.0)
        assert deviates.iloc[0]


class TestArmA:
    def test_a_quote_that_agrees_with_its_median_is_served_raw(self, monkeypatch):
        """The arm. 2% off the median is not an outlier, so serving quotes the
        price a venue actually published."""
        monkeypatch.setenv("SERVE_OUTLIER_GATED_ANCHOR", "1")
        price, smoothed = _pair([10.2], [10.0])
        base, _ = ItemForecaster._serving_base_price(price, smoothed)
        assert base.iloc[0] == pytest.approx(10.2)

    def test_a_deviating_quote_is_still_smoothed(self, monkeypatch):
        """The arm narrows the wedge; it does not remove the smoothing. A 50%
        one-day print is what the substitution was motivated by."""
        monkeypatch.setenv("SERVE_OUTLIER_GATED_ANCHOR", "1")
        price, smoothed = _pair([15.0], [10.0])
        base, _ = ItemForecaster._serving_base_price(price, smoothed)
        assert base.iloc[0] == pytest.approx(10.0)

    def test_a_deviating_quote_below_the_median_is_smoothed_too(self, monkeypatch):
        """The test is on |deviation|. A gate that caught only up-spikes would
        bias the served cross-section downward."""
        monkeypatch.setenv("SERVE_OUTLIER_GATED_ANCHOR", "1")
        price, smoothed = _pair([5.0], [10.0])
        base, deviates = ItemForecaster._serving_base_price(price, smoothed)
        assert deviates.iloc[0]
        assert base.iloc[0] == pytest.approx(10.0)

    def test_items_are_gated_independently(self, monkeypatch):
        monkeypatch.setenv("SERVE_OUTLIER_GATED_ANCHOR", "1")
        price, smoothed = _pair([10.2, 15.0, 9.9], [10.0, 10.0, 10.0])
        base, deviates = ItemForecaster._serving_base_price(price, smoothed)
        assert list(deviates) == [False, True, False]
        assert list(base) == pytest.approx([10.2, 10.0, 9.9])


class TestTheBoundary:
    def test_exactly_the_tolerance_is_not_an_outlier(self, monkeypatch):
        """The shipped mask is `> 0.10`, not `>=`. The arm must inherit that
        exactly: a boundary that moved with the flag would put items in one
        arm's raw cohort and the other's smoothed one for a reason that is not
        the arm."""
        monkeypatch.setenv("SERVE_OUTLIER_GATED_ANCHOR", "1")
        price, smoothed = _pair([11.0], [10.0])  # exactly +10%
        base, deviates = ItemForecaster._serving_base_price(price, smoothed)
        assert not deviates.iloc[0]
        assert base.iloc[0] == pytest.approx(11.0)

    def test_just_past_the_tolerance_is_an_outlier(self, monkeypatch):
        monkeypatch.setenv("SERVE_OUTLIER_GATED_ANCHOR", "1")
        price, smoothed = _pair([11.01], [10.0])
        base, deviates = ItemForecaster._serving_base_price(price, smoothed)
        assert deviates.iloc[0]
        assert base.iloc[0] == pytest.approx(10.0)

    def test_the_tolerance_is_the_ten_percent_the_log_message_claims(self):
        assert pytest.approx(0.10) == ItemForecaster.ANCHOR_OUTLIER_TOLERANCE


class TestDegradeNeverDrop:
    def test_an_item_with_no_smoothed_price_keeps_its_raw_quote(self, monkeypatch):
        """`_smoothed_anchor_prices` falls back to the latest observation, so a
        NaN here means the item was not in the frame at all. Serving cannot drop
        it, and NaN in the base price would void its whole forecast."""
        for arm in ("0", "1"):
            monkeypatch.setenv("SERVE_OUTLIER_GATED_ANCHOR", arm)
            price, smoothed = _pair([7.0], [float("nan")])
            base, deviates = ItemForecaster._serving_base_price(price, smoothed)
            assert base.iloc[0] == pytest.approx(7.0), arm
            assert not deviates.iloc[0], arm

    def test_a_zero_smoothed_price_is_not_treated_as_a_deviation(self, monkeypatch):
        """Dividing by it is the bug the `> 0` guard exists for; the shipped
        arm would also serve the zero. Under the gate the raw quote survives."""
        monkeypatch.setenv("SERVE_OUTLIER_GATED_ANCHOR", "1")
        price, smoothed = _pair([7.0], [0.0])
        base, deviates = ItemForecaster._serving_base_price(price, smoothed)
        assert not deviates.iloc[0]
        assert base.iloc[0] == pytest.approx(7.0)


class TestTheMaskIsArmInvariant:
    def test_both_arms_flag_the_same_items(self, monkeypatch):
        """The `n items >10% from 3d median` warning is how a run reports the
        size of the deviating cohort, and the replay's gate is read on that
        cohort. If the arm moved the mask, the two dispatches would describe
        different populations."""
        price, smoothed = _pair([10.2, 15.0, 5.0, 11.0], [10.0] * 4)

        monkeypatch.setenv("SERVE_OUTLIER_GATED_ANCHOR", "0")
        _, control = ItemForecaster._serving_base_price(price, smoothed)
        monkeypatch.setenv("SERVE_OUTLIER_GATED_ANCHOR", "1")
        _, arm = ItemForecaster._serving_base_price(price, smoothed)

        assert list(control) == list(arm) == [False, True, True, False]


class TestItIsWiredIntoServing:
    def test_predict_quotes_through_the_helper(self):
        """The arm has exactly one seam -- `latest_rows["price"]` after the
        substitution is the only site that sets the served `current_price`. An
        extraction that predict() does not call would leave the flag inert and
        the two dispatches identical, which reads as a null result."""

        src = method_closure_source(ItemForecaster, "predict")
        assert "_serving_base_price(" in src

    def test_the_flag_is_logged_so_a_run_says_which_arm_it_is(self):
        """A serving-only flag leaves no trace in `meta.json`: the artifact is
        the same one either way. The log line is the only record of which price
        a stored replay number was scored against."""

        src = method_closure_source(ItemForecaster, "predict")
        assert "SERVE_OUTLIER_GATED_ANCHOR" in src
