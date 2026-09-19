"""Unit tests for the archive-basis served-rank confirmation leg.

Covers the three rules the pre-registration fixes ahead of data
(`docs/research/2026-09-13-archive-basis-centre-rank-preregistration.md`):
the +-2 day forward anchor, the whole-DAY frozen exclusion (never row-level),
and the absolute VOID bar below 20 qualifying dates.
"""

import datetime as dt
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from scripts.archive_basis_centre_rank import (
    MIN_DATES,
    archive_legs,
    evaluate_bars,
    frozen_anchor_dates,
    resolve_forward_anchor,
)


def D(s):
    return dt.date.fromisoformat(s)


class TestResolveForwardAnchor:
    """Rule 1: nearest available non-frozen day within +-2, ties -> earlier."""

    def test_exact_day_wins_when_available(self):
        avail = {D("2026-08-27"), D("2026-08-28"), D("2026-08-29")}
        assert resolve_forward_anchor(D("2026-08-28"), avail, set()) == D("2026-08-28")

    def test_tie_breaks_toward_the_earlier_day(self):
        # 08-28 missing; 08-27 and 08-29 are both one day away.
        avail = {D("2026-08-27"), D("2026-08-29")}
        assert resolve_forward_anchor(D("2026-08-28"), avail, set()) == D("2026-08-27")

    def test_nearest_wins_over_earlier(self):
        avail = {D("2026-08-26"), D("2026-08-29")}
        # 08-29 is 1 away, 08-26 is 2 away -> nearest, not earliest.
        assert resolve_forward_anchor(D("2026-08-28"), avail, set()) == D("2026-08-29")

    def test_beyond_tolerance_is_unresolvable(self):
        avail = {D("2026-08-25"), D("2026-08-31")}
        assert resolve_forward_anchor(D("2026-08-28"), avail, set()) is None

    def test_frozen_day_is_skipped_even_when_exact(self):
        avail = {D("2026-08-23"), D("2026-08-24")}
        frozen = {D("2026-08-23")}
        assert resolve_forward_anchor(D("2026-08-23"), avail, frozen) == D("2026-08-24")

    def test_all_candidates_frozen_is_unresolvable(self):
        avail = {D("2026-08-22"), D("2026-08-23")}
        assert resolve_forward_anchor(D("2026-08-23"), avail, avail) is None


class TestFrozenAnchorDates:
    """Rule 2: whole days are excluded, and only on consecutive-day evidence."""

    @staticmethod
    def _frame(rows):
        df = pd.DataFrame(rows, columns=["item_id", "date", "price"])
        df["date"] = pd.to_datetime(df["date"])
        return df

    def test_flags_a_day_above_threshold(self):
        # day 3: 3 of 4 items unchanged from day 2 -> 0.75.
        rows = []
        for i, (p2, p3) in enumerate([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.5)]):
            rows += [(f"i{i}", "2026-08-02", p2), (f"i{i}", "2026-08-03", p3)]
        out = frozen_anchor_dates(self._frame(rows), threshold=0.70)
        assert out == {D("2026-08-03")}

    def test_below_threshold_is_not_flagged(self):
        rows = []
        for i, (p2, p3) in enumerate([(1.0, 1.0), (2.0, 2.0), (3.0, 3.3), (4.0, 4.5)]):
            rows += [(f"i{i}", "2026-08-02", p2), (f"i{i}", "2026-08-03", p3)]
        assert frozen_anchor_dates(self._frame(rows), threshold=0.70) == set()

    def test_non_consecutive_days_carry_no_evidence(self):
        # A gap means "unchanged" says nothing about a daily freeze.
        rows = [("i0", "2026-08-02", 1.0), ("i0", "2026-08-05", 1.0)]
        assert frozen_anchor_dates(self._frame(rows), threshold=0.70) == set()

    def test_threshold_is_inclusive(self):
        rows = []
        for i, (p2, p3) in enumerate([(1.0, 1.0), (2.0, 2.4)]):
            rows += [(f"i{i}", "2026-08-02", p2), (f"i{i}", "2026-08-03", p3)]
        assert frozen_anchor_dates(self._frame(rows), threshold=0.50) == {D("2026-08-03")}


class TestArchiveLegs:
    """All three legs share the archive anchor; rows missing any leg drop
    from BOTH arms."""

    @staticmethod
    def _prices():
        # two items, daily 08-10..08-26
        rows = []
        for i, base in enumerate([10.0, 20.0]):
            for k in range(17):
                day = D("2026-08-10") + dt.timedelta(days=k)
                rows.append((f"i{i}", day, base * (1.0 + 0.01 * k)))
        return {(s, d): p for s, d, p in rows}

    def _panel(self):
        return pd.DataFrame(
            {
                "forecast_date": [D("2026-08-12"), D("2026-08-12")],
                "slug": ["i0", "i1"],
                "predicted_price_mid": [10.5, 21.0],
            }
        )

    def test_all_three_legs_use_the_archive_anchor(self):
        px = self._prices()
        out = archive_legs(self._panel(), px, horizon=14, available=sorted({d for _, d in px}), frozen=set())
        assert len(out) == 2
        row = out.iloc[0]
        anchor = px[("i0", D("2026-08-12"))]
        fwd = px[("i0", D("2026-08-26"))]
        assert row["r_hat"] == pytest.approx(10.5 / anchor - 1)
        assert row["realized"] == pytest.approx(fwd / anchor - 1)
        prev = px[("i0", D("2026-08-11"))]
        assert row["naive"] == pytest.approx(-np.log(anchor / prev))

    def test_row_missing_any_leg_drops_entirely(self):
        px = self._prices()
        del px[("i1", D("2026-08-26"))]  # i1 loses its forward leg
        del px[("i1", D("2026-08-24"))]  # ... and both tolerated fallbacks
        del px[("i1", D("2026-08-25"))]
        out = archive_legs(self._panel(), px, horizon=14, available=sorted({d for _, d in px}), frozen=set())
        assert out["slug"].tolist() == ["i0"]

    def test_naive_requires_a_consecutive_prior_day(self):
        px = self._prices()
        del px[("i0", D("2026-08-11"))]
        out = archive_legs(self._panel(), px, horizon=14, available=sorted({d for _, d in px}), frozen=set())
        assert out["slug"].tolist() == ["i1"]

    def test_forward_offset_is_recorded(self):
        px = self._prices()
        avail = sorted({d for _, d in px} - {D("2026-08-26")})
        out = archive_legs(self._panel(), px, horizon=14, available=avail, frozen=set())
        # 08-26 gone -> tie between 08-25 and (absent) 08-27 -> earlier.
        assert set(out["fwd_offset"]) == {-1}


class TestEvaluateBars:
    """Rule 3: VOID is absolute below MIN_DATES - no statistic is read."""

    POS: ClassVar[dict] = {"ci_low": 0.05, "ci_high": 0.14, "mean": 0.09}
    SPANS: ClassVar[dict] = {"ci_low": -0.02, "ci_high": 0.11, "mean": 0.04}

    def test_void_below_min_dates_even_when_both_cis_are_positive(self):
        v = evaluate_bars(self.POS, self.POS, n_dates=MIN_DATES - 1)
        assert v["verdict"] == "VOID"
        assert "model_ic" not in v

    def test_confirmed_needs_both_cis_positive(self):
        v = evaluate_bars(self.POS, self.POS, n_dates=MIN_DATES)
        assert v["verdict"] == "CONFIRMED"

    def test_kill_when_model_ci_spans_zero(self):
        assert evaluate_bars(self.SPANS, self.POS, n_dates=MIN_DATES)["verdict"] == "KILL"

    def test_kill_when_paired_ci_is_not_positive(self):
        assert evaluate_bars(self.POS, self.SPANS, n_dates=MIN_DATES)["verdict"] == "KILL"

    def test_missing_ci_is_void_not_kill(self):
        assert evaluate_bars(None, self.POS, n_dates=MIN_DATES)["verdict"] == "VOID"
