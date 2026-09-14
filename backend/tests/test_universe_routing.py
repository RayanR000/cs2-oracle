import inspect
from unittest.mock import MagicMock

import pytest
from models.forecaster import ItemForecaster


@pytest.fixture
def fc(tmp_path):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


def test_resolve_slugs_accepts_universe():
    sig = inspect.signature(ItemForecaster._resolve_backfilled_slugs)
    assert "universe" in sig.parameters


def test_fetch_and_build_thread_universe():
    assert "universe" in inspect.signature(ItemForecaster.fetch_price_history).parameters
    assert "universe" in inspect.signature(ItemForecaster.build_training_data).parameters


def test_universe_selects_column():
    # Serve reads is_backfilled from the DB; train is derived from the archive
    # (the data repo), not an is_trainable DB column that drifted out of the
    # managed Postgres. So the resolver routes train to the archive helper.
    src = inspect.getsource(ItemForecaster._resolve_backfilled_slugs)
    assert "is_backfilled" in src
    assert "_archive_universe_slugs" in src


def test_cache_key_includes_universe():
    src = inspect.getsource(ItemForecaster._voted_cache_key)
    assert "universe" in src


# ---------------------------------------------------------------------------
# Behavioral tests: the introspection tests above pass even if `universe`
# were plumbed through but never actually used. These exercise the real
# routing logic instead of just checking signatures/source text.
# ---------------------------------------------------------------------------


def test_cache_key_differs_between_train_and_serve(fc):
    train_key = fc._voted_cache_key(days_back=1, backfilled_only=True, backfilled_slugs=None, universe="train")
    serve_key = fc._voted_cache_key(days_back=1, backfilled_only=True, backfilled_slugs=None, universe="serve")
    assert train_key != serve_key


class _CapturingDB:
    """Fake db_session that records the SQL text passed to execute()."""

    def __init__(self):
        self.captured_sql = None

    def execute(self, sql, *args, **kwargs):
        self.captured_sql = str(sql)
        return self

    def fetchall(self):
        return []


def test_resolve_slugs_train_universe_reads_archive_not_db(fc):
    # Train derives from the archive, so it must NOT query the items table
    # (the managed Postgres has no is_trainable column; a DB read there would
    # throw and fall back to loading every slug).
    capturing_db = _CapturingDB()
    fc.db = capturing_db
    from unittest.mock import patch

    with patch.object(ItemForecaster, "_archive_universe_slugs", return_value={"a", "b"}) as m:
        result = fc._resolve_backfilled_slugs(universe="train")
    assert result == {"a", "b"}
    m.assert_called_once_with(exclude_iflow=True)
    assert capturing_db.captured_sql is None


def test_resolve_slugs_routes_to_is_backfilled_for_serve_universe(fc):
    capturing_db = _CapturingDB()
    fc.db = capturing_db
    fc._resolve_backfilled_slugs(universe="serve")
    assert "is_backfilled" in capturing_db.captured_sql
    assert "is_trainable" not in capturing_db.captured_sql
