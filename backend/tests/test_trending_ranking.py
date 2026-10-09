"""The trending list must not rank by confidence.

``_build_trending`` ordered by ``desc(confidence_order)`` first, so the
anti-predictive high-confidence subset occupied the top of the list before
predicted return was considered at all. The ``"medium"`` tier it ranked between
high and low has never been emitted once — 0 rows in the entire forecast
history.

See docs/specs/2026-08-03-served-forecast-surface-design.md.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import api.routes.items as items_mod


class TestTrendingDoesNotRankByConfidence:
    def test_confidence_order_is_gone(self):
        source = inspect.getsource(items_mod._build_trending)
        assert "confidence_order" not in source

    def test_dead_medium_tier_is_gone(self):
        source = Path(items_mod.__file__).read_text()
        assert '"medium"' not in source

    def test_trending_applies_the_price_floor(self):
        source = inspect.getsource(items_mod._build_trending)
        assert "price_floor_clause" in source

    def test_trending_freshness_is_a_window_not_an_exact_today(self):
        """`forecast_date` is the day the band was anchored on -- the newest
        archived day, usually today-1 because the dump lands ~22:00 UTC -- not
        the wall clock of the run. A `== today` freshness pin empties the
        trending list after every normal daily run; the guard has to be a
        bounded recency window."""
        source = inspect.getsource(items_mod._build_trending)
        assert "forecast_date == today" not in source, (
            "an exact-today pin goes empty once forecast_date is the anchor day"
        )
        assert "forecast_date >=" in source, "the freshness guard must be a bounded window, not dropped entirely"


class TestTrendingRows:
    """Behaviour on a real (SQLite) session, not the source text."""

    @staticmethod
    def _db(forecasts):
        from datetime import date

        from database import Base, Item, ItemForecast
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool

        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine)()
        for i in range(1, 5):
            db.add(Item(id=i, item_id=f"i{i}", name=f"Item {i}", type="skin", icon_url="x", is_backfilled=1))
        for item_id, mid, cur in forecasts:
            db.add(
                ItemForecast(
                    item_id=item_id, forecast_date=date.today(), horizon_days=7, price_mid=mid, current_price=cur
                )
            )
        db.commit()
        return db

    def test_items_without_a_forecast_are_not_listed(self):
        """Item 3 and 4 have no h=7 forecast. On Postgres their NULL ratio
        sorted first under the old outer join and filled the whole list."""
        db = self._db([(1, 12.0, 10.0), (2, 21.0, 20.0)])
        rows = items_mod._build_trending(db, 10)
        assert [r.id for r in rows] == [1, 2]

    def test_ranked_by_predicted_ratio_and_priced_from_the_forecast(self):
        """No price_history rows exist, as in prod since 2026-07-11: the price
        comes from the forecast's quote, so no row is dropped for lacking one."""
        db = self._db([(1, 10.5, 10.0), (2, 30.0, 20.0), (4, 5.0, 5.0)])
        rows = items_mod._build_trending(db, 2)
        assert [r.id for r in rows] == [2, 1]
        assert [r.latest_price for r in rows] == [20.0, 10.0]
