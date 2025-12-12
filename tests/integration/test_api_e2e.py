import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.ml.dqn_agent import YassirPricingAgent
import os

MODELS_DIR = "models"
ZONE_NAME = "BAB_EZZOUAR"

@pytest.fixture(scope="function")
def test_client_with_metrics_reset():
    """
    Provides a TestClient and ensures that custom metrics are reset
    before each test function runs.
    """
    with TestClient(app) as client:
        # Reset metrics before yielding the client to the test
        client.post("/api/private/reset-metrics")
        yield client

@pytest.fixture(scope="session", autouse=True)
def create_test_model():
    """Create a versioned model for the entire test session."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    for f in os.listdir(MODELS_DIR):
        os.remove(os.path.join(MODELS_DIR, f))
    zone_config = {"max_drivers": 150, "max_requests": 300}
    test_agent = YassirPricingAgent(zone_config=zone_config)
    model_path = os.path.join(MODELS_DIR, f"{ZONE_NAME}-v20230101010000.pth")
    test_agent.save_model(model_path)

def test_ab_test_routing_and_metrics(test_client_with_metrics_reset):
    """Verify that the A/B test correctly routes users and labels metrics."""
    client = test_client_with_metrics_reset
    user_dqn_group = "user_id_abc"    # hash(...) % 10 = 2 -> DQN
    user_static_group = "user_id_123" # hash(...) % 10 = 1 -> Static

    payload = { "zone": ZONE_NAME, "hour": 8, "day_of_week": 1, "active_drivers": 60,
                "pending_requests": 150, "traffic_index": 0.6, "weather_score": 0.9 }

    # Request for DQN group
    client.post("/api/predict", json=payload, headers={"X-User-ID": user_dqn_group})

    # Request for Static group
    client.post("/api/predict", json=payload, headers={"X-User-ID": user_static_group})

    # Scrape metrics and verify counts
    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    metrics_text = metrics_response.text

    expected_dqn_metric = f'price_predictions_total{{agent_type="dqn",zone="{ZONE_NAME}"}} 1.0'
    expected_static_metric = f'price_predictions_total{{agent_type="static",zone="{ZONE_NAME}"}} 1.0'

    assert expected_dqn_metric in metrics_text
    assert expected_static_metric in metrics_text

def test_health_check_ok(test_client_with_metrics_reset):
    """Test that the API is healthy."""
    response = test_client_with_metrics_reset.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
