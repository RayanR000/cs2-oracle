"""Tests for Evidently-based data drift detection."""

import numpy as np
import pandas as pd


class TestDriftDetection:
    """Verify drift detection on synthetic data."""

    def test_no_drift_on_identical_data(self, tmp_path):
        from monitoring.drift import detect_drift

        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "price": rng.normal(100, 10, 1000),
            "return_1d": rng.normal(0, 0.02, 1000),
            "volume": rng.poisson(50, 1000).astype(float),
        })

        report = detect_drift(
            reference=df,
            current=df,
            output_dir=str(tmp_path),
        )
        assert not report.drift_detected
        assert report.n_drifted_features == 0

    def test_detects_mean_shift(self, tmp_path):
        from monitoring.drift import detect_drift

        rng = np.random.default_rng(42)
        ref = pd.DataFrame({
            "price": rng.normal(100, 10, 1000),
            "return_1d": rng.normal(0, 0.02, 1000),
        })
        cur = pd.DataFrame({
            "price": rng.normal(200, 10, 1000),
            "return_1d": rng.normal(0.1, 0.02, 1000),
        })

        report = detect_drift(
            reference=ref,
            current=cur,
            output_dir=str(tmp_path),
        )
        assert report.drift_detected
        assert report.n_drifted_features >= 1

    def test_report_files_written(self, tmp_path):
        import os

        from monitoring.drift import detect_drift

        rng = np.random.default_rng(42)
        df = pd.DataFrame({"price": rng.normal(100, 10, 500)})

        report = detect_drift(
            reference=df,
            current=df,
            output_dir=str(tmp_path),
        )
        assert os.path.exists(report.json_path)
        assert os.path.exists(report.html_path)

    def test_per_feature_scores(self, tmp_path):
        from monitoring.drift import detect_drift

        rng = np.random.default_rng(42)
        ref = pd.DataFrame({
            "stable": rng.normal(0, 1, 1000),
            "drifted": rng.normal(0, 1, 1000),
        })
        cur = pd.DataFrame({
            "stable": rng.normal(0, 1, 1000),
            "drifted": rng.normal(10, 1, 1000),
        })

        report = detect_drift(reference=ref, current=cur, output_dir=str(tmp_path))
        assert "drifted" in report.drifted_features
