"""Tests for evaluation.mlflow_logging — MLflow metric logging helper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestLogIntervalMetrics:
    def test_logs_flat_metrics(self):
        """Flat numeric metrics are logged via mlflow.log_metric."""
        mlflow = pytest.importorskip("mlflow")
        from evaluation.mlflow_logging import log_interval_metrics

        metrics = {
            "coverage": 0.85,
            "average_width": 12.5,
            "relative_width": 0.15,
            "interval_score": 18.3,
            "adaptivity": {"width_vs_abs_error": 0.45, "width_vs_sq_error": 0.38},
        }
        with patch("mlflow.log_metric") as mock_log:
            log_interval_metrics(metrics)
            calls = {c.args[0]: c.args[1] for c in mock_log.call_args_list}
            assert calls["coverage"] == 0.85
            assert calls["average_width"] == 12.5
            assert calls["interval_score"] == 18.3

    def test_logs_adaptivity_sub_keys(self):
        mlflow = pytest.importorskip("mlflow")
        from evaluation.mlflow_logging import log_interval_metrics

        metrics = {
            "coverage": 0.80,
            "average_width": 10.0,
            "relative_width": 0.10,
            "interval_score": 15.0,
            "adaptivity": {"width_vs_abs_error": 0.5, "width_vs_sq_error": 0.4},
        }
        with patch("mlflow.log_metric") as mock_log:
            log_interval_metrics(metrics)
            calls = {c.args[0]: c.args[1] for c in mock_log.call_args_list}
            assert calls["adaptivity.width_vs_abs_error"] == 0.5
            assert calls["adaptivity.width_vs_sq_error"] == 0.4

    def test_logs_with_prefix(self):
        mlflow = pytest.importorskip("mlflow")
        from evaluation.mlflow_logging import log_interval_metrics

        metrics = {
            "coverage": 0.80,
            "average_width": 10.0,
            "relative_width": 0.10,
            "interval_score": 15.0,
            "adaptivity": {"width_vs_abs_error": 0.5, "width_vs_sq_error": 0.4},
        }
        with patch("mlflow.log_metric") as mock_log:
            log_interval_metrics(metrics, prefix="h7_")
            calls = {c.args[0]: c.args[1] for c in mock_log.call_args_list}
            assert "h7_coverage" in calls
            assert "h7_interval_score" in calls

    def test_logs_coverage_by_tier(self):
        mlflow = pytest.importorskip("mlflow")
        from evaluation.mlflow_logging import log_interval_metrics

        metrics = {
            "coverage": 0.80,
            "average_width": 10.0,
            "relative_width": 0.10,
            "interval_score": 15.0,
            "adaptivity": {"width_vs_abs_error": 0.5, "width_vs_sq_error": 0.4},
            "coverage_by_tier": {0: 0.75, 1: 0.82, 2: 0.90},
        }
        with patch("mlflow.log_metric") as mock_log:
            log_interval_metrics(metrics)
            calls = {c.args[0]: c.args[1] for c in mock_log.call_args_list}
            assert calls["coverage_by_tier.0"] == 0.75
            assert calls["coverage_by_tier.1"] == 0.82

    def test_skips_non_numeric(self):
        """Non-numeric values (like calibration curve lists) are skipped."""
        mlflow = pytest.importorskip("mlflow")
        from evaluation.mlflow_logging import log_interval_metrics

        metrics = {
            "coverage": 0.80,
            "average_width": 10.0,
            "relative_width": 0.10,
            "interval_score": 15.0,
            "adaptivity": {"width_vs_abs_error": 0.5, "width_vs_sq_error": 0.4},
            "calibration_curve": [(0.5, 0.48), (0.8, 0.79)],
        }
        with patch("mlflow.log_metric") as mock_log:
            log_interval_metrics(metrics)
            logged_keys = {c.args[0] for c in mock_log.call_args_list}
            assert "calibration_curve" not in logged_keys

    def test_nan_values_skipped(self):
        """NaN metrics should be skipped."""
        mlflow = pytest.importorskip("mlflow")
        from evaluation.mlflow_logging import log_interval_metrics

        metrics = {
            "coverage": float("nan"),
            "average_width": 10.0,
            "relative_width": 0.10,
            "interval_score": 15.0,
            "adaptivity": {"width_vs_abs_error": float("nan"), "width_vs_sq_error": 0.4},
        }
        with patch("mlflow.log_metric") as mock_log:
            log_interval_metrics(metrics)
            logged_keys = {c.args[0] for c in mock_log.call_args_list}
            assert "coverage" not in logged_keys
            assert "adaptivity.width_vs_abs_error" not in logged_keys
            assert "average_width" in logged_keys
