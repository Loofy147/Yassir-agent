import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.ml.dqn_agent import YassirPricingAgent
import os
import time

MODELS_DIR = "models"
ZONE_NAME = "BAB_EZZOUAR"

@pytest.fixture(scope="session")
def test_client():
    """Create a TestClient that has had its startup events executed."""
    with TestClient(app) as client:
        yield client

@pytest.fixture(scope="session", autouse=True)
def create_versioned_test_models():
    """Create versioned models for testing the loading and metrics logic."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # Clean up old models
    for f in os.listdir(MODELS_DIR):
        os.remove(os.path.join(MODELS_DIR, f))

    zone_config = {"max_drivers": 150, "max_requests": 300}
    test_agent = YassirPricingAgent(zone_config=zone_config)

    # Create two model versions to test the loading logic
    test_agent.save_model(os.path.join(MODELS_DIR, f"{ZONE_NAME}-v20230101000000.pth"))
    test_agent.save_model(os.path.join(MODELS_DIR, f"{ZONE_NAME}-v20230101010000.pth"))

def test_loads_latest_model_version(test_client):
    """Verify that the API loads the newest model version."""
    response = test_client.get("/api/versions")
    assert response.status_code == 200
    loaded_versions = response.json()
    assert loaded_versions.get(ZONE_NAME) == f"{ZONE_NAME}-v20230101010000.pth"

def test_metrics_endpoint_standard_and_custom(test_client):
    """
    Verify the /metrics endpoint is working and tracks both standard
    and custom metrics correctly.
    """
    # 1. Make a prediction request to generate metrics
    payload = { "zone": ZONE_NAME, "hour": 10, "day_of_week": 1, "active_drivers": 50,
                "pending_requests": 100, "traffic_index": 0.5, "weather_score": 0.8 }
    response = test_client.post("/api/predict", json=payload)
    assert response.status_code == 200

    # 2. Scrape the /metrics endpoint
    metrics_response = test_client.get("/metrics")
    assert metrics_response.status_code == 200
    metrics_text = metrics_response.text

    # 3. Verify standard metric from the instrumentator
    # Checks that a request to the /api/predict endpoint was recorded
    assert f'http_requests_total{{handler="/api/predict",method="POST",status="2xx"}}' in metrics_text

    # 4. Verify custom prediction counter
    # Checks that our custom counter was incremented for the correct zone
    expected_custom_metric = f'price_predictions_total{{zone="{ZONE_NAME}"}} 1.0'
    assert expected_custom_metric in metrics_text, \
        f"Custom metric not found or incorrect. Expected: '{expected_custom_metric}'"

def test_health_check_ok(test_client):
    """Test that the API is healthy."""
    response = test_client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
