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

from datetime import date, timedelta

import pytest

from backtest.resolution_gate import (
    MAX_UNRESOLVABLE_PCT,
    classify_archive_gap,
    classify_base_gap,
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


class TestLegWindowAboveTheStalenessBound:
    """The horizon-wide count is blind past the resolver's staleness bound.

    Run 34351633071 (2026-09-09): the 2026-08-28..09-05 collection holes left
    exactly 2 covered days in the 7-day window the h=30 actual leg may draw
    from, while the 30-day horizon still held 23-24. Whole-horizon counting
    read all 16,626 as FRESH and failed the run at 14.3% over a hole no item
    could resolve through — at most 2 observations existed globally where
    ``resolve_anchors`` requires 3 within 7 days of the anchor. Counting only
    the leg-effective range (``f < day <= target`` with ``day >= target -
    staleness``) classifies them as the collection gap they are.
    """

    HOLES = {
        date(2026, 8, 28), date(2026, 8, 30), date(2026, 8, 31),
        date(2026, 9, 1), date(2026, 9, 3), date(2026, 9, 4),
        date(2026, 9, 5),
    }

    def _covered(self, f_date, target_date):
        days = set()
        d = f_date
        while d <= target_date:
            days.add(d)
            d = date.fromordinal(d.toordinal() + 1)
        return days - self.HOLES

    def test_hole_at_the_target_end_of_a_long_horizon_is_a_gap(self):
        """h=30 dated 08-05, target 09-04: 24 days in the horizon, 2 in the leg."""
        f_date, target = date(2026, 8, 5), date(2026, 9, 4)
        covered = self._covered(f_date, target)
        assert len([d for d in covered if f_date < d <= target]) == 24
        assert classify_archive_gap(
            f_date=f_date, target_date=target, covered_days=covered,
            staleness_days=7,
        )

    def test_the_legacy_count_misses_it(self):
        """Without the bound the same row reads FRESH — the 2026-09-09 failure."""
        f_date, target = date(2026, 8, 5), date(2026, 9, 4)
        assert not classify_archive_gap(
            f_date=f_date, target_date=target,
            covered_days=self._covered(f_date, target),
        )

    def test_a_full_leg_window_is_not_a_gap_at_any_horizon(self):
        """h=30 dated 08-09, target 09-08: 4 leg days, resolves in production."""
        f_date, target = date(2026, 8, 9), date(2026, 9, 8)
        assert not classify_archive_gap(
            f_date=f_date, target_date=target,
            covered_days=self._covered(f_date, target),
            staleness_days=7,
        )

    def test_exactly_a_window_in_the_leg_is_not_a_gap(self):
        """h=30 dated 08-04, target 09-03: leg holds {08-27, 08-29, 09-02}.

        The staleness floor is inclusive — resolve_anchors keeps an observation
        exactly MAX_WINDOW_SPAN_DAYS back — so this cohort resolves and must
        not be excused.
        """
        f_date, target = date(2026, 8, 4), date(2026, 9, 3)
        covered = self._covered(f_date, target)
        leg = sorted(d for d in covered
                     if f_date < d <= target and d >= date(2026, 8, 27))
        assert leg == [date(2026, 8, 27), date(2026, 8, 29), date(2026, 9, 2)]
        assert not classify_archive_gap(
            f_date=f_date, target_date=target, covered_days=covered,
            staleness_days=7,
        )

    def test_a_hole_far_from_the_target_is_not_a_gap(self):
        """h=30 missing one day 25 days out: 29 present, leg full — not excused."""
        f_date, target = date(2026, 8, 5), date(2026, 9, 4)
        covered = {f_date + timedelta(days=i) for i in range(1, 31)}
        covered.discard(date(2026, 8, 11))
        assert not classify_archive_gap(
            f_date=f_date, target_date=target, covered_days=covered,
            staleness_days=7,
        )

    def test_short_horizons_are_unchanged_by_the_bound(self):
        """Below the bound the leg range IS the horizon, so the parameter is a no-op."""
        covered = {date(2026, 7, 31), date(2026, 8, 1), date(2026, 8, 4)}
        assert classify_archive_gap(
            f_date=date(2026, 8, 1), target_date=date(2026, 8, 4),
            covered_days=covered, staleness_days=7,
        )
        full = {date(2026, 7, d) for d in range(20, 32)}
        assert not classify_archive_gap(
            f_date=date(2026, 7, 25), target_date=date(2026, 7, 28),
            covered_days=full, staleness_days=7,
        )


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


class TestBaseLegGap:
    """The gap classifier was blind to the base leg's backward window.

    Run 34691910482 (2026-09-12): "resolution rate: 5,536 of 27,680
    newly-resolvable forecasts (20.0%)". All 5,536 are the h=3 cohort dated
    09-06 — the first forecast date after the 08-28..09-05 collection holes.
    Reproduced read-only against prod with the full h=3 group load (105,374
    anchors / 99,817 resolved, matching CI exactly): the cell drops 5,536 of
    5,536 on ``base_none`` while every one of its target anchors resolves,
    because the base window [08-30, 09-06] holds exactly {09-02, 09-06} where
    ``resolve_anchors`` requires 3 observations within 7 days (08-29 sits 8
    back). The actual-leg check reads clean — 09-07/08/09 are all present — so
    the failure counted FRESH. It is permanent and deterministic: history is
    fixed, so the cell re-enters ``to_resolve`` forever and taxes every future
    fresh rate until dilution. No item could resolve through the hole, so it is
    the same collection gap, on the other leg.
    """

    HOLES = {
        date(2026, 8, 28), date(2026, 8, 30), date(2026, 8, 31),
        date(2026, 9, 1), date(2026, 9, 3), date(2026, 9, 4),
        date(2026, 9, 5),
    }

    def _covered(self, start, end):
        days = set()
        d = start
        while d <= end:
            days.add(d)
            d = date.fromordinal(d.toordinal() + 1)
        return days - self.HOLES

    def test_first_date_after_the_hole_is_a_base_gap(self):
        """h=3 dated 09-06: the base window holds {09-02, 09-06}, 2 where 3
        are required. Pins the production shape, not a toy range."""
        covered = self._covered(date(2026, 8, 20), date(2026, 9, 11))
        base = sorted(
            d for d in covered
            if date(2026, 8, 30) <= d <= date(2026, 9, 6)
        )
        assert base == [date(2026, 9, 2), date(2026, 9, 6)]
        assert classify_base_gap(
            f_date=date(2026, 9, 6), covered_days=covered,
            window=3, staleness_days=7,
        )

    def test_the_actual_leg_check_misses_it(self):
        """09-07/08/09 are all present, so the old code reads FRESH — the
        2026-09-12 failure mode."""
        assert not classify_archive_gap(
            f_date=date(2026, 9, 6), target_date=date(2026, 9, 9),
            covered_days=self._covered(date(2026, 8, 20), date(2026, 9, 11)),
            staleness_days=7,
        )

    def test_second_date_after_the_hole_is_not_a_base_gap(self):
        """h=3 dated 09-07 resolves in production: [08-31, 09-07] holds
        {09-02, 09-06, 09-07}."""
        assert not classify_base_gap(
            f_date=date(2026, 9, 7),
            covered_days=self._covered(date(2026, 8, 20), date(2026, 9, 11)),
            window=3, staleness_days=7,
        )

    def test_the_staleness_floor_is_inclusive(self):
        """An observation exactly staleness_days back still backs the leg —
        resolve_anchors drops only on strict ``>``. f=09-09, floor 09-02."""
        assert not classify_base_gap(
            f_date=date(2026, 9, 9),
            covered_days={date(2026, 9, 2), date(2026, 9, 8), date(2026, 9, 9)},
            window=3, staleness_days=7,
        )

    def test_the_forecast_date_itself_counts(self):
        """The base leg draws at or before f, f included."""
        assert not classify_base_gap(
            f_date=date(2026, 9, 9),
            covered_days={date(2026, 9, 7), date(2026, 9, 8), date(2026, 9, 9)},
            window=3, staleness_days=7,
        )

    def test_no_coverage_information_is_not_a_gap(self):
        """An empty covered set means we do not know; do not excuse the row."""
        assert not classify_base_gap(
            f_date=date(2026, 9, 6), covered_days=set(),
            window=3, staleness_days=7,
        )

    def test_a_missing_f_date_is_not_a_gap(self):
        assert not classify_base_gap(
            f_date=None, covered_days={date(2026, 9, 6)},
            window=3, staleness_days=7,
        )

    def test_the_failing_run_numbers_fail_without_the_fix(self):
        """Run 34691910482 as the old code counted it: 5,536 fresh of 27,680
        informative attempts (20.0%) over a 1.1% coverage — fatal."""
        r = evaluate_gate(
            n_mature=526208,
            n_attempted=88828,
            n_unresolvable_fresh=5536,
            n_unresolvable_chronic=210,
            n_unresolvable_gap=60938,
        )
        assert not r.ok
        assert "resolution rate" in r.reason

    def test_the_same_run_passes_once_the_cell_is_excused(self):
        """The 09-06 cell moves fresh -> gap: 0 fresh failures over 22,144
        informative attempts, 66,474 gap warned on, 210 chronic. Green."""
        r = evaluate_gate(
            n_mature=526208,
            n_attempted=88828,
            n_unresolvable_fresh=0,
            n_unresolvable_chronic=210,
            n_unresolvable_gap=66474,
        )
        assert r.ok, r.reason
        assert r.warn
        assert r.fresh_rate_pct == pytest.approx(0.0)
        assert "66,474" in r.reason
