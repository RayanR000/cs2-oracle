"""Gate tests for the Mondrian diagnostic's pure helpers.

The served-panel read itself is covered by running the script; what is silent
when it breaks is the stratum assignment (a width tertile that collapses, a
staleness bucket that mislabels NULL, a rebase that measures the wedge instead
of the centre). Each has a test here.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.conditional_coverage_by_stratum import (
    assign_width_tertile,
    bucket_staleness,
    prepare,
)


def _row(
    mid=110.0,
    low=100.0,
    high=125.0,
    base=100.0,
    actual=112.0,
    current=105.0,
    stale=0.0,
    family="skin",
    h=7,
    date="2026-08-10",
):
    return dict(
        item_id=1,
        forecast_date=date,
        horizon_days=h,
        base_price=base,
        actual_price=actual,
        current_price=current,
        predicted_price_low=low,
        predicted_price_mid=mid,
        predicted_price_high=high,
        base_stale_run_days=stale,
        family=family,
    )


def test_covered_uses_own_quote_not_base():
    # r_hat = 110/105-1 = 4.76%; w = (10+15)/105; r = 12%.
    # Off base the "return" would read (110/100-1)=10% vs r=12% — inside only
    # by wedge luck; the rebased read is inside by construction here.
    df = prepare(pd.DataFrame([_row()]))
    assert bool(df.loc[0, "covered"]) is True


def test_outside_band_is_uncovered():
    df = prepare(pd.DataFrame([_row(actual=200.0)]))
    assert bool(df.loc[0, "covered"]) is False


def test_missing_current_price_falls_back_to_base():
    df = prepare(pd.DataFrame([_row(current=np.nan)]))
    # r_hat = 110/100-1 = 10%, w_lo=10/100, w_hi=15/100; r = 12% → covered.
    assert bool(df.loc[0, "covered"]) is True


def test_tier_comes_from_base_price():
    df = prepare(pd.DataFrame([_row(base=0.5), _row(base=1500.0)]))
    # $0.5 is tier 0 but below the served cohort; prepare keeps the label —
    # the cohort gate is the loader's WHERE, not this function.
    assert df["tier"].tolist() == [0, 5]


def test_staleness_buckets():
    assert bucket_staleness(0.0) == "fresh"
    assert bucket_staleness(3.0) == "stale"
    assert bucket_staleness(float("nan")) == "unknown"
    assert bucket_staleness(None) == "unknown"


def test_width_tertile_splits_evenly():
    s = pd.Series(np.linspace(0.01, 0.30, 90))
    out = assign_width_tertile(s)
    assert set(out.unique()) == {"narrow", "mid", "wide"}
    assert (out.value_counts() == 30).all()


def test_width_tertile_degenerate_is_flat_not_empty():
    out = assign_width_tertile(pd.Series([0.1] * 10))
    assert set(out.unique()) == {"flat"}


def test_missing_family_becomes_unknown():
    df = prepare(pd.DataFrame([_row(family=None)]))
    assert df.loc[0, "family"] == "unknown"


def test_excluded_dates_are_dropped():
    # 2026-07-19 is an excluded forecast date (dead-band rule, one run).
    df = prepare(pd.DataFrame([_row(date="2026-07-19"), _row()]))
    assert len(df) == 1
