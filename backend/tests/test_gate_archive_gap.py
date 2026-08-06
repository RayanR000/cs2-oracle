"""A hole in the archive is not cohort shrinkage, and must not read as one.

The 2026-08-02/03 collection outage left `prices-2026-08.parquet` holding only
2026-08-01 and 2026-08-04. Every horizon-3 forecast whose actual leg spans that
hole is permanently unscoreable: `resolve_anchors` takes the last SMOOTH_WINDOW
(3) observations at or before the anchor and is allowed to reach
MAX_WINDOW_SPAN_DAYS (7) back, so for a forecast dated 08-01 targeting 08-04 the
window is {07-31, 08-01, 08-04} and `oldest_observation <= f_date` trips the
disjoint-leg guard. 17,382 forecasts died that way in run 30901398468, plus the
225 known chronic stragglers = the 17,607 the gate reported.

Neither existing category fits. They are not FRESH — no resolver or collector
regression is happening now, and the run's own resolution succeeded (34,764 of
34,764 anchors resolved). They are not CHRONIC either, because `classify_chronic`
asks whether coverage has moved 7 days past target_date, and on 08-04 it had not.
So they landed in the fatal coverage ratio and pinned the gate at 17.9% — a
failure that could not clear itself, since a forecast that never resolves never
earns a frozen outcome and re-enters `to_resolve` every run forever.

Backfill is not available: CSGOTrader serves only `/latest/`, and CSMarketAPI's
per-item history endpoint costs 1 request/item against a 4,000/month quota for
41,294 items.

So GAP is a third category: reported and warned on every run, excluded from the
fatal ratio, and attributed to the specific missing days so it stays actionable
instead of becoming a silent excuse.
"""
from __future__ import annotations

from datetime import date

import pytest

from backtest.resolution_gate import (
    MAX_UNRESOLVABLE_PCT,
    classify_archive_gap,
    evaluate_gate,
)


class TestTheLiveBreakage:
    def test_the_exact_0804_numbers_pass(self):
        """Run 30901398468. 17,382 gap + 225 chronic out of 98,119."""
        r = evaluate_gate(
            n_mature=98119,
            n_attempted=23149,
            n_unresolvable_fresh=0,
            n_unresolvable_chronic=225,
            n_unresolvable_gap=17382,
        )
        assert r.ok, r.reason
        assert r.warn

    def test_the_reason_names_the_gap_and_the_scoreable_cohort(self):
        r = evaluate_gate(
            n_mature=98119,
            n_attempted=23149,
            n_unresolvable_fresh=0,
            n_unresolvable_chronic=225,
            n_unresolvable_gap=17382,
        )
        assert "17,382" in r.reason
        assert "gap" in r.reason.lower()
        # 98,119 - 17,382 unscoreable
        assert "80,737" in r.reason

    def test_gap_rows_are_kept_out_of_the_fatal_coverage_ratio(self):
        """Coverage must reflect the 225 chronic, not the 17,607 total."""
        r = evaluate_gate(
            n_mature=98119,
            n_attempted=23149,
            n_unresolvable_fresh=0,
            n_unresolvable_chronic=225,
            n_unresolvable_gap=17382,
        )
        assert r.coverage_pct == pytest.approx(225 / 98119 * 100)
        assert r.gap_pct == pytest.approx(17382 / 98119 * 100)


class TestTheWarningDoesNotClaimTheScoredDenominator:
    """Run 31057993603 (2026-08-05) said "Reporting on the 80,737 scoreable
    forecasts" and then scored 66,279 of them.

    98,119 mature - 17,382 gap = 80,737, which is what this gate owns. Below it,
    225 chronic rows never earn an outcome row at all (80,512 considered) and
    14,233 frozen rows with a NULL base_price are dropped by
    `_records_from_frozen_outcomes` (66,279 scored). The gate is pure and runs
    before scoring, so it cannot see either drop — and must therefore not
    present its own survivor count as the metric's denominator.
    """

    ARGS = dict(
        n_mature=98119,
        n_attempted=23149,
        n_unresolvable_fresh=0,
        n_unresolvable_chronic=225,
        n_unresolvable_gap=17382,
    )

    def test_it_does_not_announce_what_it_is_reporting_on(self):
        r = evaluate_gate(**self.ARGS)
        assert "reporting on" not in r.reason.lower()

    def test_the_survivor_count_is_labelled_an_upper_bound(self):
        r = evaluate_gate(**self.ARGS)
        assert "80,737" in r.reason
        assert "upper bound" in r.reason.lower()

    def test_it_points_at_the_scored_count_as_the_real_denominator(self):
        """The operator has to be told where the honest number is, or the
        reframing just removes information."""
        r = evaluate_gate(**self.ARGS)
        assert "scored" in r.reason.lower()


class TestTheLoopholeIsClosed:
    def test_a_fresh_regression_still_fails_behind_a_large_gap(self):
        """A gap population must not become cover for a broken resolver.

        5,000 fresh failures out of 5,225 informative attempts is 95.7%; the
        17,382 gap rows must not dilute that away.
        """
        r = evaluate_gate(
            n_mature=98119,
            n_attempted=23149,
            n_unresolvable_fresh=5000,
            n_unresolvable_chronic=225,
            n_unresolvable_gap=17382,
        )
        assert not r.ok
        assert "resolution rate" in r.reason

    def test_gap_attempts_are_excluded_from_the_fresh_denominator(self):
        """Gap attempts carried no information, so they leave the fresh rate.

        attempted 23,149 - 225 chronic - 17,382 gap = 5,542 informative.
        """
        r = evaluate_gate(
            n_mature=98119,
            n_attempted=23149,
            n_unresolvable_fresh=100,
            n_unresolvable_chronic=225,
            n_unresolvable_gap=17382,
        )
        assert r.fresh_rate_pct == pytest.approx(100 / 5542 * 100)

    def test_a_gap_swallowing_the_whole_cohort_still_fails(self):
        """At some point there is no metric left to report, gap or not."""
        r = evaluate_gate(
            n_mature=1000,
            n_attempted=1000,
            n_unresolvable_fresh=0,
            n_unresolvable_chronic=0,
            n_unresolvable_gap=1000,
        )
        assert not r.ok
        assert "nothing scoreable" in r.reason.lower()


class TestBackwardCompatibility:
    def test_the_frozen_cohort_case_is_unchanged(self):
        """The 08-03 fix must keep working with no gap rows."""
        r = evaluate_gate(
            n_mature=75195,
            n_attempted=225,
            n_unresolvable_fresh=0,
            n_unresolvable_chronic=225,
        )
        assert r.ok, r.reason
        assert r.gap_pct is None or r.gap_pct == 0


class TestClassifyArchiveGap:
    COVERED = {date(2026, 7, 31), date(2026, 8, 1), date(2026, 8, 4)}

    def test_window_spanning_a_missing_day_is_a_gap(self):
        """h=3 dated 08-01, target 08-04: 08-02 and 08-03 are absent."""
        assert classify_archive_gap(
            f_date=date(2026, 8, 1),
            target_date=date(2026, 8, 4),
            covered_days=self.COVERED,
        )

    def test_a_fully_covered_window_is_not_a_gap(self):
        covered = {date(2026, 7, d) for d in range(20, 32)}
        assert not classify_archive_gap(
            f_date=date(2026, 7, 25),
            target_date=date(2026, 7, 28),
            covered_days=covered,
        )

    def test_the_forecast_date_itself_being_absent_is_not_counted(self):
        """Only the actual leg's window (f, target] matters for disjointness."""
        covered = {date(2026, 8, 2), date(2026, 8, 3), date(2026, 8, 4)}
        assert not classify_archive_gap(
            f_date=date(2026, 8, 1),
            target_date=date(2026, 8, 4),
            covered_days=covered,
        )

    def test_no_coverage_information_is_not_a_gap(self):
        """An empty covered set means we do not know; do not excuse the row."""
        assert not classify_archive_gap(
            f_date=date(2026, 8, 1),
            target_date=date(2026, 8, 4),
            covered_days=set(),
        )
