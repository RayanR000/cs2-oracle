"""The unresolvable gate must survive a frozen cohort without going blind.

Two failure modes pull in opposite directions, and the gate has to hold both:

1. HYPERSENSITIVITY (the live bug). The gate divided by the forecasts
   *requiring* resolution. Once the 2026-08-01 backfill froze the cohort, the
   only rows still requiring resolution were the 225 chronically-unresolvable
   stragglers the backfill itself had logged as "225 (0.4%)". They are by
   definition 100% of themselves, so the daily run failed at
   "100.0% of mature forecasts could not be resolved" on 225 rows out of 75,195
   — 0.3% of the real cohort. Guaranteed to recur every single day.

2. DILUTION (why the original denominator was chosen). Divide by the whole
   mature cohort and a genuinely broken resolver drowns: 500 fresh failures
   against 75,000 frozen rows is 0.7%, under any sane cap.

Count cannot separate the two — 225 chronic and 500 fresh failures look
identical. Chronicity can: has the archive already moved past the target date?
"""
from __future__ import annotations

from datetime import date

import pytest

from backtest.resolution_gate import (
    MAX_UNRESOLVABLE_PCT,
    classify_chronic,
    evaluate_gate,
)


class TestTheLiveBug:
    def test_frozen_cohort_with_chronic_stragglers_passes(self):
        """The exact 2026-08-03 numbers. Must not fail."""
        r = evaluate_gate(
            n_mature=75195, n_attempted=225,
            n_unresolvable_fresh=0, n_unresolvable_chronic=225,
        )
        assert r.ok, r.reason
        assert r.coverage_pct == pytest.approx(225 / 75195 * 100)

    def test_chronic_rows_do_not_create_a_fresh_rate_at_all(self):
        """225 chronic attempts leave zero informative attempts, not a 100% rate."""
        r = evaluate_gate(
            n_mature=75195, n_attempted=225,
            n_unresolvable_fresh=0, n_unresolvable_chronic=225,
        )
        assert r.fresh_rate_pct is None

    def test_the_chronic_population_is_still_surfaced_loudly(self):
        r = evaluate_gate(
            n_mature=75195, n_attempted=225,
            n_unresolvable_fresh=0, n_unresolvable_chronic=225,
        )
        assert r.ok and r.warn
        assert "225" in r.reason


class TestDilutionIsStillCaught:
    def test_a_broken_resolver_on_real_volume_still_fails(self):
        """500 fresh attempts all failing, against a large frozen cohort.

        Coverage is only 0.7%, so the coverage gate alone would pass this. The
        fresh-rate gate is what must catch it — with no attempt floor to slip
        under.
        """
        r = evaluate_gate(
            n_mature=75195, n_attempted=500, n_unresolvable_fresh=500,
        )
        assert not r.ok
        assert "resolution rate" in r.reason.lower()

    def test_a_broken_resolver_is_caught_even_alongside_chronic_rows(self):
        """The chronic tail must not dilute the fresh signal either."""
        r = evaluate_gate(
            n_mature=75195, n_attempted=725,
            n_unresolvable_fresh=500, n_unresolvable_chronic=225,
        )
        assert not r.ok
        assert "resolution rate" in r.reason.lower()

    def test_a_small_fresh_breakage_is_not_excused_by_being_small(self):
        """20 of 20 fresh failures is 100% and must fail. Count is not the test."""
        r = evaluate_gate(
            n_mature=75195, n_attempted=20, n_unresolvable_fresh=20,
        )
        assert not r.ok

    def test_mass_shrinkage_fails_on_coverage(self):
        """The 2026-08-01 shape: 30,859 of 80,737 mature and unresolvable."""
        r = evaluate_gate(
            n_mature=80737, n_attempted=80737, n_unresolvable_fresh=30859,
        )
        assert not r.ok
        assert r.coverage_pct > MAX_UNRESOLVABLE_PCT

    def test_a_chronic_population_large_enough_to_matter_still_fails(self):
        """Chronic rows are excused from the rate, never from coverage."""
        r = evaluate_gate(
            n_mature=1000, n_attempted=500,
            n_unresolvable_fresh=0, n_unresolvable_chronic=500,
        )
        assert not r.ok
        assert r.coverage_pct == pytest.approx(50.0)


class TestBoundaries:
    def test_healthy_run_passes_clean(self):
        r = evaluate_gate(
            n_mature=10000, n_attempted=1000, n_unresolvable_fresh=20,
        )
        assert r.ok and not r.warn

    def test_nothing_attempted_is_not_a_division(self):
        r = evaluate_gate(n_mature=75195, n_attempted=0, n_unresolvable_fresh=0)
        assert r.ok
        assert r.fresh_rate_pct is None

    def test_empty_cohort_is_not_a_division(self):
        r = evaluate_gate(n_mature=0, n_attempted=0, n_unresolvable_fresh=0)
        assert r.ok
        assert r.coverage_pct is None

    def test_coverage_uses_the_mature_cohort_not_the_attempts(self):
        """Mutation guard: swapping the coverage denominator for n_attempted
        would make this 100% and fail the run."""
        r = evaluate_gate(
            n_mature=100000, n_attempted=100,
            n_unresolvable_fresh=0, n_unresolvable_chronic=100,
        )
        assert r.coverage_pct == pytest.approx(0.1)
        assert r.ok

    def test_a_full_cohort_backfill_still_gates_on_rate(self):
        """--reresolve attempts everything; both denominators coincide."""
        r = evaluate_gate(
            n_mature=60962, n_attempted=60962, n_unresolvable_fresh=225,
        )
        assert r.ok
        assert r.fresh_rate_pct == pytest.approx(225 / 60962 * 100)

    def test_failure_reason_names_both_numbers(self):
        r = evaluate_gate(
            n_mature=1000, n_attempted=1000, n_unresolvable_fresh=900,
        )
        assert not r.ok
        assert "900" in r.reason and "1,000" in r.reason


class TestChronicClassification:
    def test_target_well_behind_coverage_is_chronic(self):
        assert classify_chronic(date(2026, 6, 1), date(2026, 8, 1), grace_days=14)

    def test_target_at_the_coverage_edge_is_fresh(self):
        assert not classify_chronic(date(2026, 7, 30), date(2026, 8, 1), grace_days=14)

    def test_exactly_at_the_grace_boundary_is_fresh(self):
        """Strictly greater than, so the boundary day is still given the benefit."""
        assert not classify_chronic(date(2026, 7, 18), date(2026, 8, 1), grace_days=14)

    def test_one_day_past_the_boundary_is_chronic(self):
        assert classify_chronic(date(2026, 7, 17), date(2026, 8, 1), grace_days=14)

    def test_missing_dates_are_never_chronic(self):
        """Unknown provenance must not silently excuse a row from the rate gate."""
        assert not classify_chronic(None, date(2026, 8, 1), grace_days=14)
        assert not classify_chronic(date(2026, 6, 1), None, grace_days=14)
