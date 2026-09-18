"""Immutable typed outputs for the multi-head forecast graph.

Training and serving orchestration depend on these stable interfaces rather
than private methods of the 11,000-line ItemForecaster.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_float_array(values, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


@dataclass(frozen=True)
class CentrePrediction:
    name: str
    version: str
    return_pct: np.ndarray
    price: np.ndarray

    def __post_init__(self):
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.version:
            raise ValueError("version must be non-empty")
        ret = _as_float_array(self.return_pct, name="return_pct")
        price = _as_float_array(self.price, name="price")
        if ret.shape != price.shape:
            raise ValueError(f"return_pct and price must have the same length: {ret.shape} vs {price.shape}")
        if not np.all(np.isfinite(ret)):
            raise ValueError("return_pct must be finite")
        if not np.all(np.isfinite(price)):
            raise ValueError("price must be finite")
        if not np.all(price > 0):
            raise ValueError("price must be positive")
        object.__setattr__(self, "return_pct", ret)
        object.__setattr__(self, "price", price)


@dataclass(frozen=True)
class SignedIntervalOffsets:
    lower_pct: np.ndarray
    upper_pct: np.ndarray

    def __post_init__(self):
        lower = _as_float_array(self.lower_pct, name="lower_pct")
        upper = _as_float_array(self.upper_pct, name="upper_pct")
        if lower.shape != upper.shape:
            raise ValueError(f"lower_pct and upper_pct must have the same length: {lower.shape} vs {upper.shape}")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("offsets must be finite")
        if np.any(lower > upper):
            raise ValueError("lower_pct must be <= upper_pct")
        object.__setattr__(self, "lower_pct", lower)
        object.__setattr__(self, "upper_pct", upper)


@dataclass(frozen=True)
class RankingPrediction:
    name: str
    version: str
    score: np.ndarray

    def __post_init__(self):
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.version:
            raise ValueError("version must be non-empty")
        score = _as_float_array(self.score, name="score")
        if not np.all(np.isfinite(score)):
            raise ValueError("score must be finite")
        object.__setattr__(self, "score", score)
