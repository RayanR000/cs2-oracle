"""Equal-geometry interval assembly.

Applying the same signed percentage offsets to two centres produces
intervals of identical percentage geometry. This is the only supported way
to construct a candidate interval.
"""

from __future__ import annotations

import numpy as np

from models.prediction_contracts import SignedIntervalOffsets


def percentage_offsets(low: np.ndarray, mid: np.ndarray, high: np.ndarray) -> SignedIntervalOffsets:
    low_arr = np.asarray(low, dtype=float)
    mid_arr = np.asarray(mid, dtype=float)
    high_arr = np.asarray(high, dtype=float)
    if not (low_arr.shape == mid_arr.shape == high_arr.shape):
        raise ValueError("low, mid, high must have the same length")
    if not np.all(np.isfinite(low_arr) & np.isfinite(mid_arr) & np.isfinite(high_arr)):
        raise ValueError("low, mid, high must be finite")
    if np.any(mid_arr <= 0):
        raise ValueError("mid must be positive")
    lower_pct = (low_arr / mid_arr - 1.0) * 100.0
    upper_pct = (high_arr / mid_arr - 1.0) * 100.0
    return SignedIntervalOffsets(lower_pct=lower_pct, upper_pct=upper_pct)


def assemble_interval(
    centre: np.ndarray, offsets: SignedIntervalOffsets
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centre_arr = np.asarray(centre, dtype=float)
    lower = np.asarray(offsets.lower_pct, dtype=float)
    upper = np.asarray(offsets.upper_pct, dtype=float)
    if centre_arr.shape != lower.shape or centre_arr.shape != upper.shape:
        raise ValueError("centre and offsets must have the same length")
    if not np.all(np.isfinite(centre_arr)):
        raise ValueError("centre must be finite")
    if np.any(centre_arr <= 0):
        raise ValueError("centre must be positive")
    if np.any(lower > upper):
        raise ValueError("lower_pct must be <= upper_pct")
    low = centre_arr * (1.0 + lower / 100.0)
    mid = centre_arr.copy()
    high = centre_arr * (1.0 + upper / 100.0)
    return low, mid, high
