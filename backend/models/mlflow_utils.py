"""MLflow tracking utilities for the training pipeline.

Gated by MLFLOW_ENABLED=1. When off, every function is a no-op.
"""

import json
import logging
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_ENABLED = None


def mlflow_enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("MLFLOW_ENABLED") == "1"
    return _ENABLED


def _reset():
    """Test helper: re-read the env flag on next call."""
    global _ENABLED
    _ENABLED = None


@contextmanager
def training_run(experiment_name: str = "cs2-oracle", params: dict | None = None):
    if not mlflow_enabled():
        yield None
        return

    import mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        if params:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        yield run


def log_horizon_metrics(horizon: int, metrics: dict):
    if not mlflow_enabled():
        return
    import mlflow

    flat = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            flat[f"{k}_{horizon}d"] = v
    mlflow.log_metrics(flat)


def log_artifact_file(path: str):
    if not mlflow_enabled():
        return
    import mlflow

    mlflow.log_artifact(path)


def log_meta_json(meta: dict, tmp_dir: str):
    if not mlflow_enabled():
        return
    import mlflow

    path = os.path.join(tmp_dir, "meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, default=str)
    mlflow.log_artifact(path)
