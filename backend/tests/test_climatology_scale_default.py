"""The climatology band scale is the served default since 2026-08-19, and it must
coexist with the signed-band serving armed the same day.

`docs/superpowers/specs/2026-08-19-climatology-band-scale.md`. These guard the flag
default and the one interaction that could break the cutover: the `_calibrate_conformal`
mutual-exclusion raise (`forecaster.py`, "alternative band denominators, not layers").
The signed band is a centring / two-quantile change, NOT a scale denominator, so it is
absent from that raise and the two are orthogonal. No full train (~5 min).
"""

from __future__ import annotations

import inspect

from models.forecaster import ItemForecaster


def test_climatology_scale_is_on_by_default(monkeypatch):
    monkeypatch.delenv("CLIMATOLOGY_SCALE", raising=False)
    assert ItemForecaster.climatology_scale_enabled() is True


def test_only_explicit_zero_disables_it(monkeypatch):
    monkeypatch.setenv("CLIMATOLOGY_SCALE", "0")
    assert ItemForecaster.climatology_scale_enabled() is False
    # An unparseable value keeps it on, deliberately (mirrors the training floor).
    monkeypatch.setenv("CLIMATOLOGY_SCALE", "yes")
    assert ItemForecaster.climatology_scale_enabled() is True


def test_default_flags_do_not_trip_the_mutual_exclusion_raise(monkeypatch):
    """Under production defaults — climatology on, the other three scale
    denominators off — the `_calibrate_conformal` guard predicate is False, so a
    retrain does not die at its first calibration."""
    monkeypatch.delenv("CLIMATOLOGY_SCALE", raising=False)
    for other in ("LEARNED_SCALE", "SIGMA_EXPONENT", "EXCEEDANCE_SCALE"):
        monkeypatch.delenv(other, raising=False)

    would_raise = ItemForecaster.climatology_scale_enabled() and (
        ItemForecaster.sigma_exponent_enabled() or ItemForecaster.exceedance_scale_enabled()
    )
    assert would_raise is False


def test_signed_band_is_absent_from_the_mutual_exclusion_guard():
    """The signed band is not a fourth denominator: it is calibrated against
    whichever scale wins, so it must NOT sit in the guard that raises on combined
    scale flags. Scope the check to the climatology guard block (predicate +
    message), which is the only place a spurious signed-band clause would break
    the cutover; the wider method legitimately persists `conformal_q_lo/hi`."""
    src = inspect.getsource(ItemForecaster._calibrate_conformal)
    start = src.index("if self.climatology_scale_enabled() and (")
    guard = src[start : src.index("Pick one.", start)]
    assert "another band-scale flag" in guard  # anchored on the right block
    for signed_symbol in ("band_signed", "band_offsets", "conformal_q_lo", "conformal_q_hi"):
        assert signed_symbol not in guard
