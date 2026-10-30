"""Shared test fixtures for the backend test suite.

`_hermetic_ops_dir` and `_hermetic_archive` are autouse; the others are opt-in -- tests that need them
request them by name in their function signature.
"""

from unittest.mock import MagicMock

import pytest
from database import Base
from models.forecaster import ItemForecaster
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture(autouse=True)
def _hermetic_ops_dir(monkeypatch, tmp_path_factory):
    """Point ``db.parquet.OPS_DIR`` at an empty temp dir for every test.

    Since the Phase 1.2 dual-write, every DB writer also writes the monthly ops partitions, so
    a test that exercises a writer without redirecting ``OPS_DIR`` would land rows in the real,
    gitignored ``price-archive/ops``. Tests that set ``OPS_DIR`` themselves still win.
    """
    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path_factory.mktemp("ops"))


@pytest.fixture()
def ops_mirror(request, monkeypatch):
    """Opt-in: before every ``ops_store.connect``, copy the test's ``session`` into ``OPS_DIR``.

    Since Phase 2 the backtest reads the forecast record from the partitions; this keeps tests
    that seed SQLite through the ORM meaning what they meant. Needs a ``session`` fixture in the
    module. Tests where the two stores must differ use ``tests._ops_mirror`` directly instead.
    """
    import db.ops_store as store

    from tests._ops_mirror import mirror_session_to_ops

    session = request.getfixturevalue("session")
    real = store.connect

    def connect(root=None, tables=store.TABLES):
        mirror_session_to_ops(session, root, tables)
        return real(root, tables)

    monkeypatch.setattr(store, "connect", connect)


@pytest.fixture(autouse=True)
def _hermetic_archive(monkeypatch, tmp_path_factory):
    """Point every ItemForecaster's ``archive_dir`` at an empty temp dir.

    The default is the real, gitignored ``price-archive/``. CI has no checkout
    of it, but a local run read its sidecars on every ``engineer_features``
    call (~3 s each), so local and CI ran different code paths. Tests that
    need an archive build one and assign ``archive_dir`` after construction,
    which still wins.
    """
    original = ItemForecaster.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.archive_dir = tmp_path_factory.mktemp("price-archive")

    monkeypatch.setattr(ItemForecaster, "__init__", init)


@pytest.fixture
def mock_forecaster(tmp_path):
    """An ItemForecaster backed by a MagicMock db_session.

    ``model_dir`` points at a disposable tmp_path so that save/load tests
    never clobber the real ``models/saved_models/`` directory.
    """
    return ItemForecaster(
        db_session=MagicMock(),
        model_dir=str(tmp_path / "saved_models"),
    )


@pytest.fixture
def db_session():
    """In-memory SQLite session with the full ORM schema.

    Creates all tables from ``database.Base.metadata``, yields a session,
    and disposes the engine on teardown.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
