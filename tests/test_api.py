"""Tests for the FastAPI scoring service (src/api.py)."""
import importlib

import pytest
from fastapi.testclient import TestClient

from src import train as train_module


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    """Train a tiny model into a temp artifact, then load src.api against it."""
    monkeypatch.chdir(tmp_path)
    train_module.run(data=None, target="loss", experiment="api-test", seed=1)

    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "artifacts" / "model.joblib"))

    import src.api as api_module
    importlib.reload(api_module)
    return TestClient(api_module.app)


def test_health_does_not_require_model(monkeypatch, tmp_path):
    """/health must respond even if no model artifact exists yet."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    import src.api as api_module
    importlib.reload(api_module)
    client = TestClient(api_module.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_metadata_returns_model_info(api_client):
    resp = api_client.get("/metadata")
    assert resp.status_code == 200
    body = resp.json()
    assert body["target"] == "loss"
    assert isinstance(body["features"], list)
    assert len(body["features"]) > 0


def test_predict_valid_request(api_client):
    payload = {"age": 35, "vehicle_age": 5, "state": "MH", "channel": "agent"}
    resp = api_client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_loss"] >= 0
    assert "model_version" in body
    assert "scoring_timestamp" in body


def test_predict_missing_field(api_client):
    payload = {"age": 35, "vehicle_age": 5, "state": "MH"}  # missing channel
    resp = api_client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_invalid_type(api_client):
    payload = {"age": "not-a-number", "vehicle_age": 5, "state": "MH", "channel": "agent"}
    resp = api_client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_invalid_state_rejected(api_client):
    payload = {"age": 35, "vehicle_age": 5, "state": "ZZ", "channel": "agent"}
    resp = api_client.post("/predict", json=payload)
    assert resp.status_code == 422
