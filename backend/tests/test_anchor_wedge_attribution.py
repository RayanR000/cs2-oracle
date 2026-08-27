"""Splitting the published-vs-scored wedge into its causes.

`dollar_band_wedge.py` measured the wedge and bounded what removing it would
buy. It could not say WHY the two bases differ, because the served quote could
not be rebuilt -- which is what `ANCHOR_AUDIT` now records. This module pins the
decomposition's arithmetic, which is the part that must hold before any figure
it prints means anything.

The legs are multiplicative and they must telescope EXACTLY onto the total.
An additive split would leave a cross term that grows with the wedge, and the
wedge's whole interest is in its tail.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from anchor_wedge_attribution import CAPTURE_TOL, attribute, capture_verdict  # noqa: E402


def _rows(**cols):
    n = len(next(iter(cols.values())))
    base = {"current_price": [100.0] * n, "served_base": [100.0] * n,
            "base_now": [100.0] * n, "base_price": [100.0] * n}
    base.update(cols)
    return pd.DataFrame(base)


def test_the_legs_telescope_onto_the_total():
    """The identity is the whole contract: capture x revision x residual = total."""
    df = _rows(current_price=[100.0, 50.0, 12.5],
               served_base=[100.0, 55.0, 12.0],
               base_now=[104.0, 52.0, 18.0],
               base_price=[103.0, 61.0, 9.0])
    got = attribute(df)
    lhs = (1 + got["capture_gap"]) * (1 + got["revision"]) * (1 + got["residual"])
    np.testing.assert_allclose(lhs, 1 + got["total_wedge"], rtol=1e-12)


def test_a_pure_data_revision_lands_entirely_on_the_revision_leg():
    """The hypothesis under test: same definition, same date, moved inputs."""
    df = _rows(current_price=[100.0], served_base=[100.0],
               base_now=[107.0], base_price=[107.0])
    got = attribute(df)
    assert got["capture_gap"].iloc[0] == pytest.approx(0.0)
    assert got["revision"].iloc[0] == pytest.approx(0.07)
    assert got["residual"].iloc[0] == pytest.approx(0.0)
    assert got["total_wedge"].iloc[0] == pytest.approx(0.07)


def test_a_capture_failure_is_isolated_not_smeared():
    """If the audit did not record what was served, that must show up as its own
    leg rather than being absorbed into the revision figure -- otherwise a
    broken audit reads as a confirmed hypothesis."""
    df = _rows(current_price=[100.0], served_base=[90.0],
               base_now=[90.0], base_price=[90.0])
    got = attribute(df)
    assert got["capture_gap"].iloc[0] == pytest.approx(-0.10)
    assert got["revision"].iloc[0] == pytest.approx(0.0)
    assert got["residual"].iloc[0] == pytest.approx(0.0)


def test_non_positive_prices_are_dropped_not_propagated_as_inf():
    """A zero or missing anchor divides to inf and would dominate every median."""
    df = _rows(current_price=[100.0, 0.0, 100.0, np.nan],
               served_base=[100.0, 100.0, 0.0, 100.0],
               base_now=[100.0, 100.0, 100.0, 100.0],
               base_price=[110.0, 100.0, 100.0, 100.0])
    got = attribute(df)
    assert len(got) == 1
    assert np.isfinite(got[["capture_gap", "revision", "residual",
                            "total_wedge"]].to_numpy()).all()


# --- the gate -------------------------------------------------------------
#
# The decomposition is only readable if the audit actually captured the served
# quote. That is a precondition, not a finding, so it is reported as a verdict
# and it fails loudly.

def test_capture_passes_when_the_audit_reproduces_the_served_quote():
    df = attribute(_rows(current_price=[100.0, 20.0], served_base=[100.0, 20.0]))
    ok, _ = capture_verdict(df)
    assert ok is True


def test_capture_fails_when_the_audit_and_the_forecast_row_disagree():
    """A silent pass here would void every downstream number."""
    df = attribute(_rows(current_price=[100.0] * 10,
                         served_base=[100.0] * 5 + [100.0 * (1 + 10 * CAPTURE_TOL)] * 5))
    ok, msg = capture_verdict(df)
    assert ok is False
    assert "capture" in msg.lower()
