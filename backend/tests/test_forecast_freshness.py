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
        ok, _msg = freshness_verdict(None, TODAY, TODAY)
        assert ok is False

    def test_empty_parquet_fails(self):
        ok, _msg = freshness_verdict(TODAY, None, TODAY)
        assert ok is False

    def test_a_date_ahead_of_expected_passes(self):
        """A forecast dated tomorrow is odd but it is not staleness."""
        ok, _ = freshness_verdict(date(2026, 8, 4), date(2026, 8, 4), TODAY)
        assert ok is True


class TestExpectedDateDefault:
    """The default expected date must follow the dump boundary, not the clock.

    Run 34172143963 wrote 22,144 forecasts stamped 2026-09-06 -- the correct
    anchor for a 00:05 UTC run -- and the gate failed it demanding 2026-09-08,
    because it read `utcnow().date()` while the writer read
    `resolve_snapshot_date()`. The scheduled chain hid this: its aggregator
    cron fires at 23:00 UTC, past the boundary, so the two rules agreed
    exactly once a day.
    """

    def _expected(self, monkeypatch, argv):
        """Run main() with the DB/Parquet reads stubbed, capturing `expected`."""
        import scripts.check_forecast_freshness as mod

        seen = {}

        def _capture(db_newest, parquet_newest, expected):
            seen["expected"] = expected
            return True, "stubbed"

        monkeypatch.setattr(mod, "_db_newest", lambda: TODAY)
        monkeypatch.setattr(mod, "_parquet_newest", lambda: TODAY)
        monkeypatch.setattr(mod, "freshness_verdict", _capture)
        monkeypatch.setattr("sys.argv", ["check_forecast_freshness.py", *argv])
        assert mod.main() == 0
        return seen["expected"]

    def test_default_is_the_snapshot_day_not_the_wall_clock(self, monkeypatch):
        """Before the 22:00 boundary the snapshot day is yesterday."""
        import scripts.check_forecast_freshness as mod

        monkeypatch.setattr(mod, "resolve_snapshot_date", lambda: date(2026, 9, 6))
        assert self._expected(monkeypatch, []) == date(2026, 9, 6)

    def test_explicit_expected_date_still_wins(self, monkeypatch):
        import scripts.check_forecast_freshness as mod

        monkeypatch.setattr(mod, "resolve_snapshot_date", lambda: date(2026, 9, 6))
        got = self._expected(monkeypatch, ["--expected-date", "2026-08-03"])
        assert got == date(2026, 8, 3)
