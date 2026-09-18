"""Monitoring API routes: drift reports, model info, pipeline health."""

import json
import logging
import os
from datetime import datetime

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

_DEFAULT_DRIFT_DIR = os.path.join("data", "drift_reports")
_DEFAULT_MODEL_DIR = os.path.join("models", "saved_models")


def _drift_report_dir() -> str:
    return os.environ.get("DRIFT_REPORT_DIR", _DEFAULT_DRIFT_DIR)


def _model_dir() -> str:
    return os.environ.get("MODEL_DIR", _DEFAULT_MODEL_DIR)


@router.get("/drift")
def get_drift_report():
    """Return the latest drift detection report."""
    drift_dir = _drift_report_dir()
    summary_path = os.path.join(drift_dir, "drift_summary.json")

    if not os.path.exists(summary_path):
        return {"status": "no_report", "message": "No drift report found. Run drift detection first."}

    with open(summary_path) as f:
        report = json.load(f)

    return {
        "status": "ok",
        "report": report,
        "generated_at": datetime.fromtimestamp(os.path.getmtime(summary_path)).isoformat(),
    }


@router.get("/model-info")
def get_model_info():
    """Return current model metadata from meta.json."""
    model_dir = _model_dir()
    meta_path = os.path.join(model_dir, "meta.json")

    if not os.path.exists(meta_path):
        return {"status": "no_model", "message": "No model artifact found."}

    with open(meta_path) as f:
        meta = json.load(f)

    safe_keys = [
        "model_artifact_version", "trained_at", "climatology_scale",
        "exceedance_scale", "anomaly_gbm", "feature_native_nan",
        "naive_init_score", "conformal_calibration", "conformal_beta",
        "conformal_q_lo", "conformal_q_hi", "feature_importance",
        "cv_results", "training_window_days", "n_ensembles",
        "sigma_clip", "served_coverage_factor",
    ]
    filtered = {k: meta[k] for k in safe_keys if k in meta}

    return {"status": "ok", "model": filtered}


@router.get("/health")
def monitoring_health():
    """Pipeline health summary: model age, drift status, data freshness."""
    model_dir = _model_dir()
    meta_path = os.path.join(model_dir, "meta.json")

    health = {"status": "ok", "checks": {}}

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        trained_at = meta.get("trained_at", "unknown")
        health["checks"]["model_age"] = {"trained_at": trained_at}
    else:
        health["checks"]["model_age"] = {"trained_at": None, "warning": "No model found"}

    drift_dir = _drift_report_dir()
    summary_path = os.path.join(drift_dir, "drift_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            drift = json.load(f)
        health["checks"]["drift"] = {
            "drift_detected": drift.get("drift_detected", False),
            "n_drifted": drift.get("n_drifted_features", 0),
        }
    else:
        health["checks"]["drift"] = {"status": "no_report"}

    return health
