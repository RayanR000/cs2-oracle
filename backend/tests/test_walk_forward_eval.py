"""Tests for the walk-forward temporal backtesting module."""

from datetime import date

import numpy as np
import pandas as pd
import pytest


class TestWalkForwardSplits:
    """Verify that the walk-forward splitter produces correct temporal folds."""

    def test_splits_are_non_overlapping(self):
        from models.walk_forward_eval import generate_wf_splits

        dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
        splits = generate_wf_splits(
            dates,
            train_window_days=365,
            test_window_days=30,
            stride_days=30,
            embargo_days=43,
        )
        assert len(splits) >= 5, f"Expected >=5 folds, got {len(splits)}"

        for i, split in enumerate(splits):
            assert split.train_end < split.test_start, (
                f"Fold {i}: train_end {split.train_end} >= test_start {split.test_start}"
            )
            gap = (split.test_start - split.train_end).days
            assert gap >= 43, (
                f"Fold {i}: embargo gap {gap} < 43 days"
            )

    def test_splits_advance_monotonically(self):
        from models.walk_forward_eval import generate_wf_splits

        dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
        splits = generate_wf_splits(
            dates,
            train_window_days=365,
            test_window_days=30,
            stride_days=30,
            embargo_days=43,
        )
        for i in range(1, len(splits)):
            assert splits[i].test_start > splits[i - 1].test_start, (
                f"Fold {i} test_start does not advance"
            )

    def test_embargo_respects_horizon(self):
        """Embargo = horizon + 13 per project invariant #3."""
        from models.walk_forward_eval import generate_wf_splits

        dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
        for horizon in [3, 7, 14, 30]:
            embargo = horizon + 13
            splits = generate_wf_splits(
                dates,
                train_window_days=365,
                test_window_days=30,
                stride_days=30,
                embargo_days=embargo,
            )
            for fold in splits:
                gap = (fold.test_start - fold.train_end).days
                assert gap >= embargo


class TestIntervalScoring:
    """Verify interval scoring metrics are correct."""

    def test_perfect_coverage(self):
        from models.walk_forward_eval import score_intervals

        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lo = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        hi = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

        result = score_intervals(actual, lo, hi)
        assert result.coverage == pytest.approx(1.0)
        assert result.mean_width == pytest.approx(1.0)

    def test_zero_coverage(self):
        from models.walk_forward_eval import score_intervals

        actual = np.array([10.0, 20.0, 30.0])
        lo = np.array([0.0, 0.0, 0.0])
        hi = np.array([1.0, 1.0, 1.0])

        result = score_intervals(actual, lo, hi)
        assert result.coverage == pytest.approx(0.0)

    def test_partial_coverage(self):
        from models.walk_forward_eval import score_intervals

        actual = np.array([1.0, 5.0])
        lo = np.array([0.5, 0.5])
        hi = np.array([1.5, 1.5])

        result = score_intervals(actual, lo, hi)
        assert result.coverage == pytest.approx(0.5)

    def test_handles_nan(self):
        from models.walk_forward_eval import score_intervals

        actual = np.array([1.0, np.nan, 3.0])
        lo = np.array([0.5, 1.5, 2.5])
        hi = np.array([1.5, 2.5, 3.5])

        result = score_intervals(actual, lo, hi)
        assert result.n == 2
        assert result.coverage == pytest.approx(1.0)

    def test_empty_after_nan_filter(self):
        from models.walk_forward_eval import score_intervals

        actual = np.array([np.nan, np.nan])
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])

        result = score_intervals(actual, lo, hi)
        assert result.n == 0
        assert result.coverage == 0.0


class TestWalkForwardResult:
    def test_aggregation(self):
        from models.walk_forward_eval import FoldResult, IntervalScore, WalkForwardResult, WalkForwardSplit

        result = WalkForwardResult(horizon=7)
        for i in range(3):
            result.folds.append(FoldResult(
                fold_idx=i,
                split=WalkForwardSplit(
                    train_start=date(2024, 1, 1),
                    train_end=date(2024, 12, 31),
                    test_start=date(2025, 2, 12),
                    test_end=date(2025, 3, 14),
                    fold_idx=i,
                ),
                interval_score=IntervalScore(coverage=0.8, mean_width=10.0, n=100),
                mae=0.05,
            ))

        assert result.n_folds == 3
        assert result.mean_coverage == pytest.approx(0.8)
        assert result.mean_width == pytest.approx(10.0)
        assert result.mean_mae == pytest.approx(0.05)

    def test_empty_result(self):
        from models.walk_forward_eval import WalkForwardResult

        result = WalkForwardResult(horizon=7)
        assert result.n_folds == 0
        assert result.mean_coverage == 0.0
