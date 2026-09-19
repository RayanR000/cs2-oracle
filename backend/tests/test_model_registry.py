"""Tests for MLflow model registry helpers."""

import pytest
from models.mlflow_utils import _reset, promote_model, register_model

mlflow = pytest.importorskip("mlflow", reason="mlflow not installed (mlops extra)")


@pytest.fixture(autouse=True)
def _reset_mlflow_flag():
    _reset()
    yield
    _reset()


class TestModelRegistry:
    def test_register_model(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MLFLOW_ENABLED", "1")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlruns.db'}")
        _reset()

        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlruns.db'}")
        mlflow.set_experiment("test-registry")

        with mlflow.start_run() as run:
            mlflow.log_param("test", "true")
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=mlflow.pyfunc.PythonModel(),
            )
            run_id = run.info.run_id

        mv = register_model(run_id, "cs2-oracle-forecaster")
        assert mv is not None
        assert mv.name == "cs2-oracle-forecaster"

    def test_register_noop_when_disabled(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_ENABLED", raising=False)
        _reset()

        result = register_model("fake-run-id", "cs2-oracle-forecaster")
        assert result is None

    def test_promote_sets_alias(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MLFLOW_ENABLED", "1")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlruns.db'}")
        _reset()

        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlruns.db'}")
        mlflow.set_experiment("test-promote")

        with mlflow.start_run() as run:
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=mlflow.pyfunc.PythonModel(),
            )
            run_id = run.info.run_id

        mv = register_model(run_id, "test-promote-model")
        assert mv is not None

        result = promote_model("test-promote-model", int(mv.version), alias="production")
        assert result is True

        client = mlflow.tracking.MlflowClient(f"sqlite:///{tmp_path / 'mlruns.db'}")
        alias_mv = client.get_model_version_by_alias("test-promote-model", "production")
        assert alias_mv.version == mv.version

    def test_promote_noop_when_disabled(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_ENABLED", raising=False)
        _reset()

        result = promote_model("cs2-oracle-forecaster", 1)
        assert result is None
