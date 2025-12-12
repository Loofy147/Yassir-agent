import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.ml.dqn_agent import YassirPricingAgent
import os
import time
import torch

MODELS_DIR = "models"
ZONE_NAME = "BAB_EZZOUAR"

@pytest.fixture(scope="session")
def test_client():
    """Create a TestClient that has had its startup events executed."""
    # This context manager will trigger the startup event.
    with TestClient(app) as client:
        yield client

@pytest.fixture(scope="session", autouse=True)
def create_versioned_test_models():
    """Create two versioned models to test the loading logic."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # Clean up any old models from previous runs
    for f in os.listdir(MODELS_DIR):
        os.remove(os.path.join(MODELS_DIR, f))

    zone_config = {"max_drivers": 150, "max_requests": 300}
    test_agent = YassirPricingAgent(zone_config=zone_config)

    # Create an "older" model
    old_model_path = os.path.join(MODELS_DIR, f"{ZONE_NAME}-v20230101000000.pth")
    test_agent.save_model(old_model_path)

    # Create a "newer" model
    time.sleep(1) # Ensure timestamps are different
    new_model_path = os.path.join(MODELS_DIR, f"{ZONE_NAME}-v20230101010000.pth")
    test_agent.save_model(new_model_path)

def test_loads_latest_model_version(test_client):
    """Verify that the API loads the newest version of a model."""
    response = test_client.get("/api/versions")
    assert response.status_code == 200

    loaded_versions = response.json()
    assert ZONE_NAME in loaded_versions
    # The API should have loaded the file with the later timestamp
    assert loaded_versions[ZONE_NAME] == f"{ZONE_NAME}-v20230101010000.pth"

def test_health_check_ok(test_client):
    """Test that the API is healthy when a model is present."""
    response = test_client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert ZONE_NAME in response.json()["loaded_models"]

def test_predict_price_uses_loaded_model(test_client):
    """Test a standard prediction to ensure it works with versioned models."""
    payload = { "zone": ZONE_NAME, "hour": 18, "day_of_week": 4, "active_drivers": 40,
                "pending_requests": 200, "traffic_index": 0.85, "weather_score": 0.2 }

    response = test_client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "price_multiplier" in data
    assert 0.8 <= data["price_multiplier"] <= 2.0
