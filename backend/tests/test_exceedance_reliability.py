"""Phase D: offline reliability of the exceedance head, on the replay path.

Scores predicted `exceed_p` (Phase A) against the realised ONE-SIDED exceedance
`actual_ret > actionable_threshold(tier, csfloat)` — the SAME bar the label is
built on (`prepare_targets`), tier from the anchor quote via `scoring.price_tier`.
This is the pre-flip validation: no served `exceed_p` exists yet, so it runs
through `replay_serving`, which rebuilds forecasts from the archive — exactly how
band coverage was validated before it was confirmed on served outcomes.

Scope: docs/superpowers/plans/2026-08-16-exceedance-band-scale-phase2-plan.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from backtest.friction import actionable_threshold
from scripts.replay_serving import _reliability_ece, _reliability_rows


def _frame(exceed_p, current):
    return pd.DataFrame({"exceed_p": list(exceed_p), "current": list(current)})


def test_returns_empty_without_a_head():
    """No exceed_p column (pre-Phase-A artifact / EXCEEDANCE_HEAD off) -> skip."""
    frame = pd.DataFrame({"current": [10.0] * 10})
    assert _reliability_rows(frame, np.zeros(10)) == []


def test_returns_empty_when_all_probabilities_nan():
    frame = _frame([np.nan] * 10, [10.0] * 10)
    assert _reliability_rows(frame, np.zeros(10)) == []


def test_realised_rate_uses_the_one_sided_threshold():
    """current=10 -> tier 1. Two rows in one high-prob bin: one clears the bar,
    one falls short -> realised rate 0.5."""
    thr = actionable_threshold(1, "csfloat")
    frame = _frame([0.95, 0.95], [10.0, 10.0])
    actual_ret = np.array([thr + 0.10, thr - 0.05])
    rows = _reliability_rows(frame, actual_ret, n_bins=5)
    assert len(rows) == 1
    assert rows[0]["n"] == 2
    assert rows[0]["realized"] == pytest.approx(0.5)
    assert rows[0]["pred"] == pytest.approx(0.95)


def test_a_downside_move_never_counts_as_exceedance():
    """One-sided: a large DOWN move must not clear the upside cost bar."""
    frame = _frame([0.9], [10.0])
    rows = _reliability_rows(frame, np.array([-0.5]), n_bins=5)
    assert rows and rows[0]["realized"] == 0.0


def test_bins_are_fixed_width_over_the_unit_interval():
    """Fixed-width bins so a bin means the same thing across anchors/horizons."""
    frame = _frame([0.05, 0.15, 0.95], [10.0, 10.0, 10.0])
    rows = _reliability_rows(frame, np.zeros(3), n_bins=5)
    assert {r["bin"] for r in rows} == {0, 4}  # <0.2 and [0.8,1.0]


def test_ece_is_the_count_weighted_abs_gap():
    rows = [{"n": 1, "pred": 0.9, "realized": 0.0}, {"n": 3, "pred": 0.1, "realized": 0.0}]
    assert _reliability_ece(rows) == pytest.approx((1 * 0.9 + 3 * 0.1) / 4)


def test_ece_of_no_rows_is_nan():
    assert np.isnan(_reliability_ece([]))
