"""A green forecast run is not evidence that forecasts exist.

``item_forecasts`` holds 5 distinct forecast dates across the eight months from
2025-12-01 — 2026-07-17, 07-18, 07-19 and 07-29, four of them manual local
runs. Nothing failed; nothing checked. The same silent-success shape emptied the
supply scraper and reddit collector (docs/changelog/
2026-07-31-accuracy-work-closed.md).

Since Phase 2 PR 3 the verdict reads the ops partitions only: Postgres went read-only on
2026-10-30 and stops at forecast date 2026-10-29. Its newest date is still reported, and a
Postgres behind the partitions is a ``::notice``, not a failure.

See docs/specs/2026-08-03-served-forecast-surface-design.md.
"""

from __future__ import annotations

from datetime import date

from scripts.check_forecast_freshness import freshness_verdict, newest_forecast_date, postgres_note

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
    def test_partitions_current_passes(self):
        ok, msg = freshness_verdict(TODAY, TODAY)
        assert ok is True
        assert "2026-08-03" in msg

    def test_partitions_stale_fails(self):
        ok, msg = freshness_verdict(YESTERDAY, TODAY)
        assert ok is False
        assert "item_forecasts" in msg and "2026-08-02" in msg

    def test_empty_or_missing_partitions_fail_as_stale(self):
        ok, msg = freshness_verdict(None, TODAY)
        assert ok is False
        assert "no forecasts" in msg

    def test_a_date_ahead_of_expected_passes(self):
        """A forecast dated tomorrow is odd but it is not staleness."""
        ok, _ = freshness_verdict(date(2026, 8, 4), TODAY)
        assert ok is True


class TestPostgresNote:
    def test_postgres_level_with_the_partitions_says_nothing(self):
        assert postgres_note(TODAY, TODAY) is None

    def test_postgres_behind_the_partitions_is_a_notice(self):
        note = postgres_note(YESTERDAY, TODAY)
        assert note.startswith("::notice")
        assert "2026-08-02" in note and "2026-08-03" in note

    def test_unreadable_postgres_is_a_notice(self):
        assert postgres_note(None, TODAY).startswith("::notice")

    def test_postgres_ahead_says_nothing_here(self):
        """The partitions being behind is the verdict's job, not the note's."""
        assert postgres_note(TODAY, YESTERDAY) is None


class TestMainDoesNotGateOnPostgres:
    def _run(self, monkeypatch, store, db):
        import scripts.check_forecast_freshness as mod

        monkeypatch.setattr(mod, "_store_newest", lambda: store)
        monkeypatch.setattr(mod, "_db_newest", lambda: db)
        monkeypatch.setattr(mod, "_shadow_inputs", lambda: ([], set()))
        monkeypatch.setattr("sys.argv", ["check_forecast_freshness.py", "--expected-date", "2026-08-03"])
        return mod.main()

    def test_a_stale_postgres_passes_when_the_partitions_are_current(self, monkeypatch, capsys):
        assert self._run(monkeypatch, TODAY, date(2026, 7, 30)) == 0
        assert "::notice" in capsys.readouterr().out

    def test_current_postgres_does_not_rescue_stale_partitions(self, monkeypatch):
        assert self._run(monkeypatch, YESTERDAY, TODAY) == 1


class TestStoreReads:
    def test_newest_date_comes_from_the_partitions(self):
        import pandas as pd
        import scripts.check_forecast_freshness as mod
        from db import ops_partitions

        ops_partitions.upsert(
            "item_forecasts",
            pd.DataFrame(
                [
                    {"id": 1, "item_id": 1, "forecast_date": YESTERDAY, "horizon_days": 3},
                    {"id": 2, "item_id": 1, "forecast_date": TODAY, "horizon_days": 3},
                ]
            ),
            allow_closed=True,
        )
        assert mod._store_newest() == TODAY

    def test_missing_partitions_read_as_no_forecasts_not_a_crash(self):
        import scripts.check_forecast_freshness as mod

        assert mod._store_newest() is None

    def test_a_failed_postgres_read_is_none_not_a_crash(self, monkeypatch):
        import scripts.check_forecast_freshness as mod

        def boom():
            raise RuntimeError("DiskFull")

        monkeypatch.setattr(mod, "SessionLocal", boom)
        assert mod._db_newest() is None


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

        def _capture(store_newest, expected):
            seen["expected"] = expected
            return True, "stubbed"

        monkeypatch.setattr(mod, "_db_newest", lambda: TODAY)
        monkeypatch.setattr(mod, "_store_newest", lambda: TODAY)
        monkeypatch.setattr(mod, "_shadow_inputs", lambda: ([], set()))
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
