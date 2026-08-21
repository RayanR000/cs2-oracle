"""Pure-function tests for the volatility/stability tag derivation.

These back the API surface that exposes, per item x horizon:
  - expected_swing_pct  (half-band / mid)
  - a Stable/Moderate/Volatile label from the within-horizon swing tertiles

No forecasting here - just the arithmetic and the labelling rule, tested in
isolation so the endpoints can stay thin. See the v1 design in chat.
"""
from __future__ import annotations

import math

import pytest

from api.volatility_tags import (
    swing_pct, compute_thresholds, label_for, build_ranking, tag_fields,
    move_odds_calibrated, CALIBRATED_MOVE_ODDS_HORIZONS,
)


def _row(item_id, name, price, low, high, mid, exceed_p=None):
    return dict(item_id=item_id, name=name, current_price=price,
                low=low, high=high, mid=mid, exceed_p=exceed_p)


class TestSwingPct:
    def test_half_band_over_mid(self):
        # low=90 high=110 mid=100 -> half-band 10 / 100 = 0.10
        assert swing_pct(90.0, 110.0, 100.0) == pytest.approx(0.10)

    def test_asymmetric_band_uses_half_width(self):
        # low=80 high=120 mid=100 -> half-width (120-80)/2 = 20 / 100 = 0.20
        assert swing_pct(80.0, 120.0, 100.0) == pytest.approx(0.20)

    def test_zero_width_band_is_zero_swing(self):
        assert swing_pct(100.0, 100.0, 100.0) == 0.0

    def test_nonpositive_mid_returns_none(self):
        assert swing_pct(0.0, 10.0, 0.0) is None
        assert swing_pct(-5.0, 5.0, -1.0) is None

    def test_missing_input_returns_none(self):
        assert swing_pct(None, 110.0, 100.0) is None
        assert swing_pct(90.0, None, 100.0) is None
        assert swing_pct(90.0, 110.0, None) is None


class TestComputeThresholds:
    def test_tertile_cut_points(self):
        # 0..9: 1/3 quantile ~= 3.0, 2/3 quantile ~= 6.0 (linear interpolation)
        lo, hi = compute_thresholds([float(x) for x in range(10)])
        assert lo == pytest.approx(3.0)
        assert hi == pytest.approx(6.0)
        assert lo <= hi

    def test_ignores_none_values(self):
        lo, hi = compute_thresholds([0.0, None, 3.0, None, 6.0, 9.0])
        # same non-None distribution as a 4-point set; just assert ordering + finiteness
        assert math.isfinite(lo) and math.isfinite(hi) and lo <= hi

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_thresholds([])


class TestLabelFor:
    thresholds = (0.10, 0.20)

    def test_below_low_threshold_is_stable(self):
        assert label_for(0.05, self.thresholds) == "Stable"

    def test_between_thresholds_is_moderate(self):
        assert label_for(0.15, self.thresholds) == "Moderate"

    def test_above_high_threshold_is_volatile(self):
        assert label_for(0.30, self.thresholds) == "Volatile"

    def test_low_boundary_is_inclusive_stable(self):
        # swing == low threshold counts as Stable (<=), so the bottom tertile is closed
        assert label_for(0.10, self.thresholds) == "Stable"

    def test_high_boundary_is_moderate_not_volatile(self):
        # Volatile is strictly greater than the high threshold
        assert label_for(0.20, self.thresholds) == "Moderate"

    def test_none_swing_has_no_label(self):
        assert label_for(None, self.thresholds) is None


class TestBuildRanking:
    # five items with swings 5%,10%,15%,20%,25% (mid=100, half-band = pct*100)
    def _universe(self):
        return [
            _row("a", "A", 100.0, 95.0, 105.0, 100.0, exceed_p=0.30),   # 5%
            _row("b", "B", 100.0, 90.0, 110.0, 100.0, exceed_p=0.10),   # 10%
            _row("c", "C", 100.0, 85.0, 115.0, 100.0, exceed_p=None),   # 15%
            _row("d", "D", 100.0, 80.0, 120.0, 100.0, exceed_p=0.50),   # 20%
            _row("e", "E", 100.0, 75.0, 125.0, 100.0, exceed_p=0.05),   # 25%
        ]

    def test_computes_swing_move_odds_and_label(self):
        out = build_ranking(self._universe())
        by_id = {r["item_id"]: r for r in out}
        assert by_id["a"]["expected_swing_pct"] == pytest.approx(0.05)
        assert by_id["d"]["move_odds"] == pytest.approx(0.50)      # exceed_p passthrough
        assert by_id["c"]["move_odds"] is None                     # missing exceed_p
        # tertiles of {.05,.10,.15,.20,.25}: label ends span Stable..Volatile
        assert by_id["a"]["stability_label"] == "Stable"
        assert by_id["e"]["stability_label"] == "Volatile"

    def test_default_sort_is_swing_descending(self):
        out = build_ranking(self._universe())
        assert [r["item_id"] for r in out] == ["e", "d", "c", "b", "a"]

    def test_ascending_order(self):
        out = build_ranking(self._universe(), order="asc")
        assert [r["item_id"] for r in out] == ["a", "b", "c", "d", "e"]

    def test_sort_by_move_odds_puts_missing_last(self):
        out = build_ranking(self._universe(), sort="move_odds", order="desc")
        # d(.50), a(.30), b(.10), e(.05), then c(None) last
        assert [r["item_id"] for r in out] == ["d", "a", "b", "e", "c"]

    def test_limit_applied_after_sort(self):
        out = build_ranking(self._universe(), limit=2)
        assert [r["item_id"] for r in out] == ["e", "d"]

    def test_rows_without_a_band_are_dropped(self):
        rows = self._universe() + [_row("z", "Z", 100.0, None, None, None)]
        out = build_ranking(rows)
        assert "z" not in {r["item_id"] for r in out}

    def test_empty_universe_returns_empty(self):
        assert build_ranking([]) == []

    def test_labels_reflect_within_call_universe_not_absolute(self):
        # a universe of only calm items -> the widest is still "Volatile" relative to peers
        calm = [
            _row("p", "P", 100.0, 99.0, 101.0, 100.0),   # 1%
            _row("q", "Q", 100.0, 98.0, 102.0, 100.0),   # 2%
            _row("r", "R", 100.0, 97.0, 103.0, 100.0),   # 3%
        ]
        out = {r["item_id"]: r["stability_label"] for r in build_ranking(calm)}
        assert out["p"] == "Stable"
        assert out["r"] == "Volatile"


class TestBuildRankingLabelFilter:
    # same universe as TestBuildRanking: swings 5/10/15/20/25% ->
    # full-universe tertiles give Stable={a,b}, Moderate={c}, Volatile={d,e}
    def _universe(self):
        return [
            _row("a", "A", 100.0, 95.0, 105.0, 100.0, exceed_p=0.30),   # 5%
            _row("b", "B", 100.0, 90.0, 110.0, 100.0, exceed_p=0.10),   # 10%
            _row("c", "C", 100.0, 85.0, 115.0, 100.0, exceed_p=None),   # 15%
            _row("d", "D", 100.0, 80.0, 120.0, 100.0, exceed_p=0.50),   # 20%
            _row("e", "E", 100.0, 75.0, 125.0, 100.0, exceed_p=0.05),   # 25%
        ]

    def test_none_returns_whole_universe(self):
        assert len(build_ranking(self._universe(), label=None)) == 5

    def test_stable_returns_only_stable(self):
        out = build_ranking(self._universe(), label="Stable")
        assert {r["item_id"] for r in out} == {"a", "b"}
        assert all(r["stability_label"] == "Stable" for r in out)

    def test_volatile_returns_top_tertile_sorted(self):
        out = build_ranking(self._universe(), label="Volatile")
        # widest tertile {d,e}, default sort is swing descending
        assert [r["item_id"] for r in out] == ["e", "d"]

    def test_filter_is_on_full_universe_labels_not_a_re_tertiled_subset(self):
        # THE correctness property: the label is computed over the whole
        # universe, then filtered -- not by re-tertiling the filtered subset.
        univ = self._universe()
        full = {r["item_id"]: r["stability_label"] for r in build_ranking(univ)}
        for want in ("Stable", "Moderate", "Volatile"):
            got = {r["item_id"] for r in build_ranking(univ, label=want)}
            assert got == {i for i, lbl in full.items() if lbl == want}

    def test_limit_applies_after_label_filter(self):
        # of the two Volatile items {d,e}, limit=1 keeps the widest (e)
        out = build_ranking(self._universe(), label="Volatile", limit=1)
        assert [r["item_id"] for r in out] == ["e"]


class TestTagFields:
    thresholds = (0.10, 0.20)

    def test_full_tags(self):
        out = tag_fields(80.0, 120.0, 100.0, exceed_p=0.30, thresholds=self.thresholds)
        assert out == {"expected_swing_pct": pytest.approx(0.20),
                       "move_odds": 0.30, "stability_label": "Moderate"}

    def test_missing_exceed_p_gives_null_move_odds(self):
        out = tag_fields(90.0, 110.0, 100.0, exceed_p=None, thresholds=self.thresholds)
        assert out["move_odds"] is None
        assert out["stability_label"] == "Stable"   # 10% swing at low boundary

    def test_no_thresholds_leaves_label_none_but_keeps_swing(self):
        out = tag_fields(80.0, 120.0, 100.0, exceed_p=0.4, thresholds=None)
        assert out["expected_swing_pct"] == pytest.approx(0.20)
        assert out["move_odds"] == 0.4
        assert out["stability_label"] is None

    def test_degenerate_band_gives_all_none_swing(self):
        out = tag_fields(0.0, 10.0, 0.0, exceed_p=0.2, thresholds=self.thresholds)
        assert out["expected_swing_pct"] is None
        assert out["stability_label"] is None

    def test_move_odds_suppressed_at_uncalibrated_horizon(self):
        # exceed_p is calibrated only at h3/h7 (replay ECE <1.3pp; ~3.8pp at h30),
        # so an uncalibrated horizon must not publish the probability.
        out = tag_fields(80.0, 120.0, 100.0, exceed_p=0.30,
                         thresholds=self.thresholds, calibrated_move_odds=False)
        assert out["move_odds"] is None
        # swing and label are band-width, valid at every horizon
        assert out["expected_swing_pct"] == pytest.approx(0.20)
        assert out["stability_label"] == "Moderate"


class TestMoveOddsCalibrated:
    def test_true_for_short_horizons(self):
        assert move_odds_calibrated(3) is True
        assert move_odds_calibrated(7) is True

    def test_false_for_long_horizons(self):
        assert move_odds_calibrated(14) is False
        assert move_odds_calibrated(30) is False

    def test_calibrated_set_is_exactly_3_and_7(self):
        assert set(CALIBRATED_MOVE_ODDS_HORIZONS) == {3, 7}


class TestBuildRankingCalibration:
    def _rows(self):
        return [
            _row("a", "A", 100.0, 80.0, 120.0, 100.0, exceed_p=0.30),
            _row("b", "B", 100.0, 90.0, 110.0, 100.0, exceed_p=0.05),
        ]

    def test_move_odds_present_when_calibrated(self):
        out = {r["item_id"]: r["move_odds"] for r in
               build_ranking(self._rows(), calibrated_move_odds=True)}
        assert out == {"a": 0.30, "b": 0.05}

    def test_move_odds_nulled_when_uncalibrated(self):
        ranked = build_ranking(self._rows(), calibrated_move_odds=False)
        assert all(r["move_odds"] is None for r in ranked)
        # swing/label untouched
        assert all(r["expected_swing_pct"] is not None for r in ranked)

    def test_default_keeps_move_odds(self):
        out = {r["item_id"]: r["move_odds"] for r in build_ranking(self._rows())}
        assert out == {"a": 0.30, "b": 0.05}
