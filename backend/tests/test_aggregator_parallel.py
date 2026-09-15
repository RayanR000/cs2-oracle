# backend/tests/test_aggregator_parallel.py
from concurrent.futures import Future
from unittest.mock import patch, MagicMock
from collectors.csgotrader_aggregator import CSGOTraderAggregator

def _completed(result):
    f = Future()
    f.set_result(result)
    return f

def test_fetch_uses_parallel_execution():
    agg = CSGOTraderAggregator()
    with patch("collectors.csgotrader_aggregator.ThreadPoolExecutor") as mock_pool:
        mock_executor = MagicMock()
        mock_pool.return_value.__enter__ = MagicMock(return_value=mock_executor)
        mock_pool.return_value.__exit__ = MagicMock(return_value=False)
        # Real completed futures: as_completed() hangs forever on MagicMocks.
        mock_executor.submit.side_effect = lambda fn, *a, **k: _completed((a[0], {}))
        try:
            agg.fetch_all_market_data()
        except Exception:
            pass
        assert mock_executor.submit.call_count >= 1

def test_aggregator_context_manager():
    agg = CSGOTraderAggregator()
    assert hasattr(agg, "close") or hasattr(agg, "__exit__")
