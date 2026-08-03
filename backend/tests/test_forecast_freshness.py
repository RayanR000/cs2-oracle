"""A green forecast run is not evidence that forecasts exist.

``item_forecasts`` holds 5 distinct forecast dates across the eight months from
2025-12-01 — 2026-07-17, 07-18, 07-19 and 07-29, four of them manual local
runs. Nothing failed; nothing checked. The same silent-success shape emptied the
supply scraper and reddit collector (docs/changelog/
2026-07-31-accuracy-work-closed.md).

Both stores are asserted because the API reads Parquet first with a DB
fallback, so a DB-only write still serves nothing.

See docs/superpowers/specs/2026-08-03-served-forecast-surface-design.md.
"""
from __future__ import annotations

from datetime import date

from scripts.check_forecast_freshness import freshness_verdict, newest_forecast_date

TODAY = date(2026, 8, 3)
YESTERDAY = date(2026, 8, 2)


class TestNewestForecastDate:
    def test_picks_the_maximum(self):
        assert newest_forecast_date([YESTERDAY, TODAY]) == TODAY

    def test_ignores_none_entries(self):
        assert newest_forecast_date([None, YESTERDAY]) == YESTERDAY

    def test_empty_is_none(self):
        assert newest_forecast_date([]) is None

    def test_all_none_is_none(self):
        assert newest_forecast_date([None, None]) is None


class TestFreshnessVerdict:
    def test_both_stores_current_passes(self):
        ok, msg = freshness_verdict(TODAY, TODAY, TODAY)
        assert ok is True
        assert "2026-08-03" in msg

    def test_db_stale_fails(self):
        ok, msg = freshness_verdict(YESTERDAY, TODAY, TODAY)
        assert ok is False
        assert "item_forecasts" in msg

    def test_parquet_stale_fails_even_when_db_is_current(self):
        """The regression that matters: the API reads Parquet first."""
        ok, msg = freshness_verdict(TODAY, YESTERDAY, TODAY)
        assert ok is False
        assert "Parquet" in msg

    def test_empty_db_fails(self):
        ok, msg = freshness_verdict(None, TODAY, TODAY)
        assert ok is False

    def test_empty_parquet_fails(self):
        ok, msg = freshness_verdict(TODAY, None, TODAY)
        assert ok is False

    def test_a_date_ahead_of_expected_passes(self):
        """A forecast dated tomorrow is odd but it is not staleness."""
        ok, _ = freshness_verdict(date(2026, 8, 4), date(2026, 8, 4), TODAY)
        assert ok is True
