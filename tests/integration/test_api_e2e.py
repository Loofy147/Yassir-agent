import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.ml.dqn_agent import YassirPricingAgent # Import the agent
import os
import time
import torch # Import torch to save the model

# The TestClient doesn't automatically run startup events. We need to trigger them manually.
@pytest.fixture(scope="session")
def test_client():
    """Create a TestClient that has had its startup events executed."""
    with TestClient(app) as client:
        yield client

# This fixture now creates a VALID, minimal PyTorch model for the tests.
@pytest.fixture(scope="session", autouse=True)
def create_valid_test_model():
    """Ensures a valid, loadable model file exists so the API can start up healthily."""
    MODELS_DIR = "models"
    ZONE_NAME = "BAB_EZZOUAR"
    MODEL_PATH = os.path.join(MODELS_DIR, f"{ZONE_NAME}.pth")

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    if not os.path.exists(MODEL_PATH):
        print(f"Creating a temporary test model at: {MODEL_PATH}")
        # Create a barebones agent instance with the correct config
        zone_config = {"max_drivers": 150, "max_requests": 300}
        test_agent = YassirPricingAgent(zone_config=zone_config)
        # Save its initial (untrained) state. This is a valid torch file.
        test_agent.save_model(MODEL_PATH)

def test_health_check_ok(test_client):
    """Test that the API is healthy when a model is present."""
    response = test_client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_price_bab_ezzouar_surge(test_client):
    """Test a prediction scenario that should result in a price surge."""
    payload = {
        "zone": "BAB_EZZOUAR",
        "hour": 18,
        "day_of_week": 4,
        "active_drivers": 40,
        "pending_requests": 200,
        "traffic_index": 0.85,
        "weather_score": 0.2
    }

    response = test_client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "price_multiplier" in data
    assert 0.8 <= data["price_multiplier"] <= 2.0

def test_predict_price_bab_ezzouar_discount(test_client):
    """Test a prediction scenario that should result in a price discount."""
    payload = {
        "zone": "BAB_EZZOUAR",
        "hour": 2,
        "day_of_week": 6,
        "active_drivers": 140,
        "pending_requests": 10,
        "traffic_index": 0.1,
        "weather_score": 0.9
    }

    response = test_client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "price_multiplier" in data
    assert 0.8 <= data["price_multiplier"] <= 2.0

def test_predict_price_latency_sla(test_client):
    """Test that the prediction latency is within the 50ms SLA."""
    payload = {
        "zone": "BAB_EZZOUAR",
        "hour": 10,
        "day_of_week": 2,
        "active_drivers": 100,
        "pending_requests": 50,
        "traffic_index": 0.5,
        "weather_score": 0.8
    }

    start_time = time.time()
    response = test_client.post("/api/predict", json=payload)
    latency_ms = (time.time() - start_time) * 1000

    assert response.status_code == 200
    print(f"Prediction latency: {latency_ms:.2f}ms")
    assert latency_ms < 50, f"Latency {latency_ms:.2f}ms exceeds 50ms SLA"

def test_predict_price_zone_not_found(test_client):
    """Test the response when a model for the requested zone is not found."""
    payload = {
        "zone": "UNKNOWN_ZONE",
        "hour": 12,
        "day_of_week": 3,
        "active_drivers": 70,
        "pending_requests": 80,
        "traffic_index": 0.4,
        "weather_score": 0.7
    }

    response = test_client.post("/api/predict", json=payload)
    assert response.status_code == 404
    assert response.json()["detail"] == "Model for zone 'UNKNOWN_ZONE' not found."
