"""The opportunities surface must not select on confidence, and must not show
tick-quantised items.

Directional accuracy split by served confidence, computed within each forecast
date so both groups faced the same market, was negative in all nine
date x horizon x model cells with n_high >= 30 — high confidence scored 4-31%
against low confidence 25-55%. The surface gated on ``confidence == "high"`` in
four queries and ranked by it, so it promoted the worst forecasts first. It also
had no price floor: the newest date offered 1,680 "undervalued" candidates with
a median price of $0.50.

See docs/superpowers/specs/2026-08-03-served-forecast-surface-design.md.
"""
from __future__ import annotations

from types import SimpleNamespace

from api.routes.opportunities import opportunity_type_for, select_opportunities


def _forecast(item_id=1, direction="up", confidence="low", current=10.0, mid=11.0):
    return SimpleNamespace(
        item_id=item_id,
        direction=direction,
        confidence=confidence,
        current_price=current,
        price_mid=mid,
    )


def _item(item_id=1, name="AK-47 | Redline"):
    return SimpleNamespace(id=item_id, name=name)


def _items_map(*items):
    return {i.id: i for i in items}


class TestOpportunityType:
    def test_up_is_undervalued_without_high_confidence(self):
        assert opportunity_type_for("up") == "undervalued"

    def test_down_is_overheated_without_high_confidence(self):
        assert opportunity_type_for("down") == "overheated"

    def test_flat_is_momentum(self):
        assert opportunity_type_for("flat") == "momentum"

    def test_none_is_momentum(self):
        assert opportunity_type_for(None) == "momentum"


class TestSelectOpportunities:
    def test_low_confidence_item_is_eligible(self):
        """The regression: low confidence was excluded from undervalued."""
        results = select_opportunities(
            [_forecast(confidence="low")], _items_map(_item()), None, 10
        )
        assert [r.opportunity_type for r in results] == ["undervalued"]

    def test_sub_dollar_item_is_dropped(self):
        results = select_opportunities(
            [_forecast(current=0.03, mid=0.04)], _items_map(_item()), None, 10
        )
        assert results == []

    def test_item_exactly_at_the_floor_is_kept(self):
        results = select_opportunities(
            [_forecast(current=1.0, mid=1.2)], _items_map(_item()), None, 10
        )
        assert len(results) == 1

    def test_ranks_by_absolute_predicted_return_not_confidence(self):
        big_low_conf = _forecast(item_id=1, confidence="low", current=10.0, mid=14.0)
        small_high_conf = _forecast(item_id=2, confidence="high", current=10.0, mid=10.5)
        results = select_opportunities(
            [small_high_conf, big_low_conf],
            _items_map(_item(1), _item(2, "Glock")),
            None,
            10,
        )
        assert [r.item_id for r in results] == [1, 2]

    def test_type_filter_selects_one_bucket(self):
        results = select_opportunities(
            [_forecast(item_id=1, direction="up"), _forecast(item_id=2, direction="down", mid=9.0)],
            _items_map(_item(1), _item(2, "Glock")),
            "overheated",
            10,
        )
        assert [r.item_id for r in results] == [2]

    def test_limit_is_applied_after_sorting(self):
        forecasts = [
            _forecast(item_id=1, current=10.0, mid=10.5),
            _forecast(item_id=2, current=10.0, mid=14.0),
        ]
        results = select_opportunities(
            forecasts, _items_map(_item(1), _item(2, "Glock")), None, 1
        )
        assert [r.item_id for r in results] == [2]

    def test_missing_item_row_is_skipped(self):
        results = select_opportunities([_forecast(item_id=99)], {}, None, 10)
        assert results == []

    def test_none_direction_is_skipped(self):
        results = select_opportunities(
            [_forecast(direction=None)], _items_map(_item()), None, 10
        )
        assert results == []


class TestNoConfidenceGateRemains:
    def test_module_source_never_filters_on_confidence(self):
        """A source-level guard, because the four gates were in SQLAlchemy
        filters that cannot be exercised without a PostgreSQL connection —
        these endpoints use DISTINCT ON, which SQLite cannot compile."""
        from pathlib import Path

        import api.routes.opportunities as mod

        source = Path(mod.__file__).read_text()
        assert "ItemForecast.confidence" not in source
        assert "high confidence" not in source
