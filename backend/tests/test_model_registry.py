"""Tests for MLflow model registry helpers."""

import pytest
import mlflow
from models.mlflow_utils import _reset, register_model, promote_model


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

    def test_promote_noop_when_disabled(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_ENABLED", raising=False)
        _reset()

        result = promote_model("cs2-oracle-forecaster", 1)
        assert result is None
