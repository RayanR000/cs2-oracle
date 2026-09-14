"""The staleness axis on scored outcomes.

`base_stale_run_days` is how long the forecast-date anchor had already been
frozen when the forecast was made — the direction the Getmansky-Lo-Makarov MA(k)
mechanism actually runs. It is a finer version of the two-bucket
`actual_price == base_price` carry-forward split, which only asks whether the
two legs came out equal.

The load-bearing property here is that NULL is its own band. Every outcome
resolved before 2026-08-08 carries one, and folding those into `fresh` would
report 13 years of unmeasured rows as measured-fresh.
"""

from __future__ import annotations

import pytest
from backtest.scoring import (
    STALENESS_BANDS,
    STALENESS_UNKNOWN,
    score_by_staleness,
    score_cohort,
    staleness_band,
)


def _rec(run_days, correct=True, **kw):
    base = kw.pop("base_price", 10.0)
    actual = kw.pop("actual_price", 11.0)
    r = {
        "abs_error": 1.0,
        "pct_error": 10.0,
        "sq_error": 1.0,
        "direction_correct": correct,
        "predicted_direction": "up",
        "actual_direction": "up",
        "in_interval": True,
        "confidence": "high",
        "base_price": base,
        "actual_price": actual,
        "price_tier": 2,
        "item_id": 1,
        "forecast_date": "2026-07-01",
        "predicted_mid": 11.0,
        "horizon_days": 7,
        "base_stale_run_days": run_days,
    }
    r.update(kw)
    return r


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "fresh"),
        (1, "repeat_1"),
        (2, "run_2_6"),
        (6, "run_2_6"),
        (7, "run_7_plus"),
        (900, "run_7_plus"),
    ],
)
def test_band_boundaries(value, expected):
    assert staleness_band(value) == expected


@pytest.mark.parametrize("value", [None, "", "abc", -1])
def test_unmeasurable_values_are_unknown_never_fresh(value):
    """The whole point of the band. A NULL is not a zero."""
    assert staleness_band(value) == STALENESS_UNKNOWN


def test_every_band_is_present_even_when_empty():
    """A stored row must be self-describing: a missing key is not a zero count."""
    out = score_by_staleness([_rec(0)])
    expected = {b[0] for b in STALENESS_BANDS} | {STALENESS_UNKNOWN}
    assert set(out) == expected
    assert out["run_7_plus"]["n"] == 0
    # None, not 0.0 — an empty band has no accuracy.
    assert out["run_7_plus"]["directional_accuracy"] is None


def test_counts_and_shares_partition_the_cohort():
    records = [_rec(0), _rec(0), _rec(1), _rec(8), _rec(None)]
    out = score_by_staleness(records)
    assert out["fresh"]["n"] == 2
    assert out["repeat_1"]["n"] == 1
    assert out["run_2_6"]["n"] == 0
    assert out["run_7_plus"]["n"] == 1
    assert out[STALENESS_UNKNOWN]["n"] == 1
    assert sum(b["n"] for b in out.values()) == len(records)
    assert sum(b["share_pct"] for b in out.values()) == pytest.approx(100.0)


def test_accuracy_is_computed_within_the_band_not_pooled():
    records = [
        _rec(0, correct=True),
        _rec(0, correct=True),
        _rec(9, correct=False),
        _rec(9, correct=False),
    ]
    out = score_by_staleness(records)
    assert out["fresh"]["directional_accuracy"] == 100.0
    assert out["run_7_plus"]["directional_accuracy"] == 0.0


def test_score_cohort_publishes_the_axis():
    metrics, n = score_cohort([_rec(0), _rec(3)])
    assert n == 2
    assert metrics["staleness_bands"]["fresh"]["n"] == 1
    assert metrics["staleness_bands"]["run_2_6"]["n"] == 1


def test_records_predating_the_column_score_as_unknown_without_crashing():
    """--rescore walks rows written long before the column existed."""
    r = _rec(0)
    del r["base_stale_run_days"]
    metrics, n = score_cohort([r])
    assert n == 1
    assert metrics["staleness_bands"][STALENESS_UNKNOWN]["n"] == 1
    assert metrics["staleness_bands"]["fresh"]["n"] == 0


def test_empty_band_set_on_empty_cohort_does_not_divide_by_zero():
    assert score_by_staleness([])["fresh"]["share_pct"] == 0.0
