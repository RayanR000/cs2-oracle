import inspect
from models.forecaster import ItemForecaster

def test_resolve_slugs_accepts_universe():
    sig = inspect.signature(ItemForecaster._resolve_backfilled_slugs)
    assert "universe" in sig.parameters

def test_fetch_and_build_thread_universe():
    assert "universe" in inspect.signature(ItemForecaster.fetch_price_history).parameters
    assert "universe" in inspect.signature(ItemForecaster.build_training_data).parameters

def test_universe_selects_column():
    src = inspect.getsource(ItemForecaster._resolve_backfilled_slugs)
    assert "is_trainable" in src and "is_backfilled" in src

def test_cache_key_includes_universe():
    src = inspect.getsource(ItemForecaster._voted_cache_key)
    assert "universe" in src
