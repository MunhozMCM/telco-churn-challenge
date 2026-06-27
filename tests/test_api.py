"""API contract tests — /health and /predict via FastAPI TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import app as app_module
from src.api.app import app


@pytest.fixture
def client(trained_pipeline):
    """TestClient with a guaranteed-loaded model (independent of models/ on disk)."""
    with TestClient(app) as test_client:
        app_module.ml["model"] = trained_pipeline
        app_module.ml.setdefault("meta", {"model_version": "test"})
        yield test_client


def test_health_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_valid_contract(client, valid_payload):
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_flag"] in (0, 1)
    assert body["risk_level"] in ("High", "Low")
    assert "X-Process-Time" in resp.headers


def test_predict_rejects_invalid_category(client, valid_payload):
    bad = dict(valid_payload)
    bad["Contract"] = "Lifetime"
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_rejects_out_of_range_numeric(client, valid_payload):
    bad = dict(valid_payload)
    bad["Tenure Months"] = -3
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422
