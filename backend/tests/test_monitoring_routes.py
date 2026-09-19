"""Tests for the monitoring API routes."""

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-for-testing")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("DRIFT_REPORT_DIR", str(tmp_path / "drift"))
    from main import app
    return TestClient(app)


class TestDriftEndpoint:
    def test_drift_returns_empty_when_no_report(self, client):
        response = client.get("/monitoring/drift")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_report"

    def test_drift_returns_report_when_present(self, client, tmp_path, monkeypatch):
        report_dir = tmp_path / "drift"
        report_dir.mkdir(exist_ok=True)
        report_data = {
            "drift_detected": True,
            "n_drifted_features": 2,
            "n_total_features": 10,
            "drifted_features": ["price", "return_1d"],
        }
        (report_dir / "drift_summary.json").write_text(json.dumps(report_data))

        response = client.get("/monitoring/drift")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["report"]["drift_detected"] is True


class TestModelInfoEndpoint:
    def test_model_info_no_model(self, client):
        response = client.get("/monitoring/model-info")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_model"

    def test_model_info_returns_metadata(self, client, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir(exist_ok=True)
        meta = {
            "trained_at": "2026-09-18T00:00:00",
            "model_artifact_version": 6,
            "climatology_scale": True,
            "conformal_calibration": {"3": {"q_hat": 95.5}},
        }
        (model_dir / "meta.json").write_text(json.dumps(meta))

        response = client.get("/monitoring/model-info")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model"]["model_artifact_version"] == 6


class TestHealthEndpoint:
    def test_monitoring_health(self, client):
        response = client.get("/monitoring/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "checks" in data
