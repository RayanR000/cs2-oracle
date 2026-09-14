"""Tests for the news -> `events` sync that unblocks event correlation.

These use an in-memory SQLite session and pass it in explicitly. `backend/.env`
points at PRODUCTION Supabase and the engine binds at import, so a test that let
`sync` open its own session would be writing to prod.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from database import Base, Event  # noqa: E402
from scripts.ingest_steam_news import news_rows  # noqa: E402
from scripts.sync_events_from_news import (  # noqa: E402
    GID_MARKER,
    format_description,
    load_valve_events,
    sync,
)


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine, tables=[Event.__table__])
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _news_item(gid, ts, title, feed_type=1):
    return {
        "gid": gid,
        "date": int(ts.timestamp()),
        "title": title,
        "feed_type": feed_type,
        "url": f"https://example/{gid}",
    }


def _parquet(tmp_path, items):
    path = tmp_path / "event-news.parquet"
    news_rows(items).to_parquet(path, index=False)
    return path


VALVE = datetime(2026, 8, 1, 12, 0)
PRESS = datetime(2026, 8, 2, 12, 0)


class TestNewsRows:
    def test_keeps_identity_the_date_panel_throws_away(self):
        rows = news_rows([_news_item("g1", VALVE, "Trade Up Contract update")])
        assert rows.iloc[0]["gid"] == "g1"
        assert rows.iloc[0]["title"] == "Trade Up Contract update"
        assert rows.iloc[0]["is_valve"] == 1

    def test_cross_posts_of_one_announcement_collapse(self):
        """Steam publishes one announcement under several gids. Counting them
        separately would duplicate the `events` row for one real event."""
        rows = news_rows(
            [
                _news_item("g1", VALVE, "Same announcement"),
                _news_item("g2", VALVE, "Same announcement"),
            ]
        )
        assert len(rows) == 1

    def test_press_is_carried_but_flagged(self):
        rows = news_rows([_news_item("p1", PRESS, "Skins are up", feed_type=0)])
        assert rows.iloc[0]["is_valve"] == 0


class TestLoading:
    def test_press_is_excluded(self, tmp_path):
        path = _parquet(
            tmp_path,
            [
                _news_item("g1", VALVE, "Valve post"),
                _news_item("p1", PRESS, "Press piece", feed_type=0),
            ],
        )
        assert load_valve_events(path, None)["gid"].tolist() == ["g1"]

    def test_since_bounds_the_scan(self, tmp_path):
        path = _parquet(
            tmp_path,
            [
                _news_item("old", datetime(2015, 1, 1), "Ancient"),
                _news_item("new", VALVE, "Recent"),
            ],
        )
        assert load_valve_events(path, date(2026, 1, 1))["gid"].tolist() == ["new"]

    def test_a_missing_table_says_which_script_writes_it(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="ingest_steam_news"):
            load_valve_events(tmp_path / "nope.parquet", None)


class TestSync:
    def test_inserts_valve_announcements(self, tmp_path, db):
        path = _parquet(tmp_path, [_news_item("g1", VALVE, "Trade Up update")])
        out = sync(path, db, since=None)

        assert out["events_inserted"] == 1
        stored = db.query(Event).one()
        assert stored.type == "update"
        assert "Trade Up update" in stored.description
        assert stored.timestamp.date() == date(2026, 8, 1)

    def test_rerunning_inserts_nothing(self, tmp_path, db):
        path = _parquet(tmp_path, [_news_item("g1", VALVE, "Trade Up update")])
        sync(path, db, since=None)
        again = sync(path, db, since=None)

        assert again["events_inserted"] == 0
        assert again["already_present"] == 1
        assert db.query(Event).count() == 1

    def test_an_edited_title_does_not_duplicate_the_event(self, tmp_path, db):
        """Matching on (timestamp, description) would insert a second row every
        time Steam edits a headline. The gid is the identity."""
        sync(_parquet(tmp_path, [_news_item("g1", VALVE, "Original")]), db, since=None)
        edited = tmp_path / "edited"
        edited.mkdir()
        sync(_parquet(edited, [_news_item("g1", VALVE, "Edited headline")]), db, since=None)

        assert db.query(Event).count() == 1

    def test_dry_run_writes_nothing(self, tmp_path, db):
        path = _parquet(tmp_path, [_news_item("g1", VALVE, "Trade Up update")])
        out = sync(path, db, since=None, dry_run=True)

        assert out["would_insert"] == 1
        assert out["events_inserted"] == 0
        assert db.query(Event).count() == 0

    def test_press_never_reaches_the_events_table(self, tmp_path, db):
        path = _parquet(tmp_path, [_news_item("p1", PRESS, "Press piece", feed_type=0)])
        sync(path, db, since=None)
        assert db.query(Event).count() == 0

    def test_an_empty_feed_is_not_an_error(self, tmp_path, db):
        path = tmp_path / "event-news.parquet"
        news_rows([]).to_parquet(path, index=False)
        out = sync(path, db, since=None)
        assert out["status"] == "success"
        assert out["events_inserted"] == 0


class TestDescription:
    def test_gid_survives_a_title_longer_than_the_column(self):
        desc = format_description("x" * 900, "g1")
        assert len(desc) <= 500
        assert desc.endswith(f"{GID_MARKER}g1]")

    def test_whitespace_is_collapsed(self):
        assert format_description("a\n\n  b", "g1").startswith("a b")
