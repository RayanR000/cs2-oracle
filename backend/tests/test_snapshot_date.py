"""The snapshot date must come from the data's day, not from the clock.

The archive lost 2026-07-27, 2026-07-30 and 2026-08-03 outright, and none of
those days had a failing run. The aggregator cron is `0 23 * * *`, and Actions
scheduling delay was observed pushing actual starts anywhere from 23:52 to
00:08. Both the pipeline (`agg_date = now.strftime(...)`) and the workflow
(`--date $(date -u +%F)`, three CSV paths) read the wall clock independently, so
a delay across midnight silently relabelled the day: two consecutive firings at
23:5x stamped the same date and the day between got nothing.

Worse, the two clock reads are separate instants. A pipeline finishing at
23:59:58 writes `aggregator-snapshots-D.csv` while the append step starting at
00:00:02 looks for `aggregator-snapshots-D+1.csv` — a hard failure that had
simply not been rolled yet.

CSGOTrader regenerates its dumps once a day around ~21:40 UTC, so the day a
snapshot belongs to is decided by which dump it came from, not by when the
runner happened to wake up.
"""
from __future__ import annotations

from datetime import date, datetime

import pytest

from collectors.snapshot_date import DUMP_PUBLISHED_HOUR_UTC, resolve_snapshot_date


class TestCronDelayCannotRelabelTheDay:
    def test_run_before_midnight_stamps_that_day(self):
        """23:52 on 08-03 is 08-03's dump."""
        assert resolve_snapshot_date(
            now=datetime(2026, 8, 3, 23, 52)
        ) == date(2026, 8, 3)

    def test_run_delayed_past_midnight_still_stamps_the_dump_day(self):
        """The live bug: 00:08 on 08-04 is still 08-03's dump, not 08-04's."""
        assert resolve_snapshot_date(
            now=datetime(2026, 8, 4, 0, 8)
        ) == date(2026, 8, 3)

    def test_the_two_observed_firings_stamp_different_days(self):
        """23:57 on 08-02 and 00:08 on 08-04 must not collide.

        Under the wall clock these produced 08-02 and 08-04, leaving 08-03
        empty. The pair is ~24h apart, so they are consecutive dump days.
        """
        first = resolve_snapshot_date(now=datetime(2026, 8, 2, 23, 57))
        second = resolve_snapshot_date(now=datetime(2026, 8, 4, 0, 8))
        assert first == date(2026, 8, 2)
        assert second == date(2026, 8, 3)
        assert (second - first).days == 1


class TestBoundaryIsTheDumpNotTheCron:
    def test_manual_run_in_the_afternoon_stamps_the_previous_dump(self):
        """At 14:00 the newest dump is still yesterday's ~21:40 one."""
        assert resolve_snapshot_date(
            now=datetime(2026, 8, 4, 14, 0)
        ) == date(2026, 8, 3)

    def test_run_just_after_the_dump_publishes_stamps_today(self):
        assert resolve_snapshot_date(
            now=datetime(2026, 8, 4, DUMP_PUBLISHED_HOUR_UTC, 30)
        ) == date(2026, 8, 4)


class TestExplicitOverride:
    def test_override_wins_over_the_clock(self):
        """Manual backfill of a specific day must be possible."""
        assert resolve_snapshot_date(
            now=datetime(2026, 8, 4, 0, 8), override="2026-07-27"
        ) == date(2026, 7, 27)

    def test_blank_override_is_ignored(self):
        """An unset env var arrives as "" and must not blow up."""
        assert resolve_snapshot_date(
            now=datetime(2026, 8, 3, 23, 52), override=""
        ) == date(2026, 8, 3)

    def test_malformed_override_is_refused_loudly(self):
        with pytest.raises(ValueError):
            resolve_snapshot_date(now=datetime(2026, 8, 3, 23, 52), override="not-a-date")


class TestIdempotenceAcrossSeparateClockReads:
    def test_two_reads_either_side_of_midnight_agree(self):
        """The pipeline and the append step must resolve the same day.

        These are the instants that would have produced a filename mismatch.
        """
        pipeline_read = resolve_snapshot_date(now=datetime(2026, 8, 3, 23, 59, 58))
        append_read = resolve_snapshot_date(now=datetime(2026, 8, 4, 0, 0, 2))
        assert pipeline_read == append_read == date(2026, 8, 3)
