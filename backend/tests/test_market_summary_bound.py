"""The market summary builder must not load unbounded rows from the DB."""

from unittest.mock import MagicMock, patch

from api.routes.market import _build_market_summary


class TestMarketSummaryBound:
    def test_query_has_limit(self):
        db = MagicMock()
        query = db.query.return_value.filter.return_value.order_by.return_value
        query.limit.return_value.all.return_value = []

        _build_market_summary(db, type=None, q=None)

        query.limit.assert_called_once()
        bound = query.limit.call_args[0][0]
        assert 0 < bound <= 5000, f"limit {bound} is unreasonable"
