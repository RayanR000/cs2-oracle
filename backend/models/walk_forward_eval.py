"""Walk-forward temporal backtesting for interval forecasts.

Produces temporally honest evaluation: train on [T-W, T], predict [T+embargo, T+embargo+test],
slide T forward by stride, repeat. Each fold is logged to MLflow as a child run.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardSplit:
    """One temporal fold's date boundaries."""

    train_start: date
    train_end: date
    test_start: date
    test_end: date
    fold_idx: int


@dataclass
class IntervalScore:
    """Metrics for a set of prediction intervals."""

    coverage: float
    mean_width: float
    median_width: float = 0.0
    n: int = 0


@dataclass
class FoldResult:
    """Results from one walk-forward fold."""

    fold_idx: int
    split: WalkForwardSplit
    interval_score: IntervalScore
    mae: float = 0.0
    ic: float = 0.0


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward evaluation results."""

    horizon: int
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def mean_coverage(self) -> float:
        if not self.folds:
            return 0.0
        return float(np.mean([f.interval_score.coverage for f in self.folds]))

    @property
    def mean_width(self) -> float:
        if not self.folds:
            return 0.0
        return float(np.mean([f.interval_score.mean_width for f in self.folds]))

    @property
    def mean_mae(self) -> float:
        if not self.folds:
            return 0.0
        return float(np.mean([f.mae for f in self.folds]))

    @property
    def n_folds(self) -> int:
        return len(self.folds)


def generate_wf_splits(
    dates: pd.DatetimeIndex,
    train_window_days: int = 365,
    test_window_days: int = 30,
    stride_days: int = 30,
    embargo_days: int = 43,
) -> list[WalkForwardSplit]:
    """Generate walk-forward date splits with embargo.

    The embargo enforces invariant #3: never pass a bare horizon to a purge.
    For horizon h, the caller should pass embargo_days = h + 13.
    """
    min_date = dates.min().date() if hasattr(dates.min(), "date") else dates.min()
    max_date = dates.max().date() if hasattr(dates.max(), "date") else dates.max()

    splits = []
    fold_idx = 0
    train_start = min_date

    while True:
        train_end = train_start + timedelta(days=train_window_days)
        test_start = train_end + timedelta(days=embargo_days)
        test_end = test_start + timedelta(days=test_window_days)

        if test_end > max_date:
            break

        splits.append(
            WalkForwardSplit(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                fold_idx=fold_idx,
            )
        )
        fold_idx += 1
        train_start += timedelta(days=stride_days)

    return splits


def score_intervals(
    actual: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> IntervalScore:
    """Score prediction intervals: coverage, width, and adaptivity."""
    mask = np.isfinite(actual) & np.isfinite(lo) & np.isfinite(hi)
    actual, lo, hi = actual[mask], lo[mask], hi[mask]
    n = len(actual)
    if n == 0:
        return IntervalScore(coverage=0.0, mean_width=0.0, n=0)

    covered = (actual >= lo) & (actual <= hi)
    widths = hi - lo

    return IntervalScore(
        coverage=float(covered.mean()),
        mean_width=float(widths.mean()),
        median_width=float(np.median(widths)),
        n=n,
    )
