"""Shared test fixtures for the backend test suite.

`_hermetic_archive` is autouse; the others are opt-in -- tests that need them
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
