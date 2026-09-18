"""Tests for MLflow tracking integration."""

import json
import os

import mlflow
import pytest
from models.mlflow_utils import (
    _reset,
    log_artifact_file,
    log_horizon_metrics,
    log_meta_json,
    mlflow_enabled,
    training_run,
)


@pytest.fixture(autouse=True)
def _reset_mlflow_flag():
    _reset()
    yield
    _reset()


class TestMLflowEnabled:
    def test_enabled_when_flag_set(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_ENABLED", "1")
        _reset()
        assert mlflow_enabled() is True

    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_ENABLED", raising=False)
        _reset()
        assert mlflow_enabled() is False

    def test_disabled_when_flag_not_one(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_ENABLED", "true")
        _reset()
        assert mlflow_enabled() is False


class TestTrainingRun:
    def test_creates_run_with_params(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MLFLOW_ENABLED", "1")
        db_uri = f"sqlite:///{tmp_path / 'mlruns.db'}"
        monkeypatch.setenv("MLFLOW_TRACKING_URI", db_uri)
        _reset()

        with training_run(experiment_name="test", params={"lr": 0.01, "epochs": 10}) as run:
            assert run is not None
            run_id = run.info.run_id

        client = mlflow.tracking.MlflowClient(db_uri)
        fetched = client.get_run(run_id)
        assert fetched.data.params["lr"] == "0.01"
        assert fetched.data.params["epochs"] == "10"

    def test_noop_when_disabled(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_ENABLED", raising=False)
        _reset()

        with training_run(params={"x": 1}) as run:
            assert run is None


class TestLogHorizonMetrics:
    def test_logs_metrics(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MLFLOW_ENABLED", "1")
        db_uri = f"sqlite:///{tmp_path / 'mlruns.db'}"
        monkeypatch.setenv("MLFLOW_TRACKING_URI", db_uri)
        _reset()

        with training_run(experiment_name="test-hz") as run:
            log_horizon_metrics(7, {"mae": 0.05, "q_hat": 141.4})
            run_id = run.info.run_id

        client = mlflow.tracking.MlflowClient(db_uri)
        fetched = client.get_run(run_id)
        assert fetched.data.metrics["mae_7d"] == pytest.approx(0.05)
        assert fetched.data.metrics["q_hat_7d"] == pytest.approx(141.4)

    def test_skips_non_numeric(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MLFLOW_ENABLED", "1")
        db_uri = f"sqlite:///{tmp_path / 'mlruns.db'}"
        monkeypatch.setenv("MLFLOW_TRACKING_URI", db_uri)
        _reset()

        with training_run(experiment_name="test-skip") as run:
            log_horizon_metrics(3, {"mae": 0.1, "name": "test"})
            run_id = run.info.run_id

        client = mlflow.tracking.MlflowClient(db_uri)
        fetched = client.get_run(run_id)
        assert "mae_3d" in fetched.data.metrics
        assert "name_3d" not in fetched.data.metrics


class TestLogArtifact:
    def test_logs_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MLFLOW_ENABLED", "1")
        db_uri = f"sqlite:///{tmp_path / 'mlruns.db'}"
        monkeypatch.setenv("MLFLOW_TRACKING_URI", db_uri)
        _reset()

        artifact = tmp_path / "test_artifact.txt"
        artifact.write_text("hello")

        with training_run(experiment_name="test-art") as run:
            log_artifact_file(str(artifact))
            run_id = run.info.run_id

        client = mlflow.tracking.MlflowClient(db_uri)
        artifacts = [a.path for a in client.list_artifacts(run_id)]
        assert "test_artifact.txt" in artifacts

    def test_logs_meta_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MLFLOW_ENABLED", "1")
        db_uri = f"sqlite:///{tmp_path / 'mlruns.db'}"
        monkeypatch.setenv("MLFLOW_TRACKING_URI", db_uri)
        _reset()

        with training_run(experiment_name="test-meta") as run:
            log_meta_json({"version": 6, "trained_at": "2026-09-18"}, str(tmp_path))
            run_id = run.info.run_id

        client = mlflow.tracking.MlflowClient(db_uri)
        artifacts = [a.path for a in client.list_artifacts(run_id)]
        assert "meta.json" in artifacts
