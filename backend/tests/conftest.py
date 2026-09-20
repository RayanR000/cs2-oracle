"""Shared test fixtures for the backend test suite.

Both fixtures are opt-in (not autouse) -- tests that need them request them by
name in their function signature.
"""

from unittest.mock import MagicMock

import pytest
from database import Base
from models.forecaster import ItemForecaster
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


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
