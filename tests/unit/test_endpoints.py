import pytest
from fastapi.testclient import TestClient
from src.main import app
from unittest.mock import patch

client = TestClient(app)

def test_health_check():
    """
    Tests the health check endpoint.
    """
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_price_dqn():
    """
    Tests the predict price endpoint with the DQN agent.
    """
    payload = { "zone": "BAB_EZZOUAR", "hour": 8, "day_of_week": 1, "active_drivers": 60,
                "pending_requests": 150, "traffic_index": 0.6, "weather_score": 0.9 }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    assert "price_multiplier" in response.json()
    assert response.json()["price_multiplier"] in [0.8, 1.0, 1.3, 1.6, 2.0]

def test_predict_price_static():
    """
    Tests the predict price endpoint with the static agent.
    """
    payload = { "zone": "BAB_EZZOUAR", "hour": 8, "day_of_week": 1, "active_drivers": 60,
                "pending_requests": 150, "traffic_index": 0.6, "weather_score": 0.9 }
    response = client.post("/api/predict", json=payload, headers={"X-User-ID": "user_id_5"})
    assert response.status_code == 200
    assert "price_multiplier" in response.json()
    assert response.json()["price_multiplier"] == 1.3

def test_predict_price_fallback():
    """
    Tests the predict price endpoint with the fallback strategy.
    """
    payload = { "zone": "UNKNOWN_ZONE", "hour": 8, "day_of_week": 1, "active_drivers": 60,
                "pending_requests": 150, "traffic_index": 0.6, "weather_score": 0.9 }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    assert "price_multiplier" in response.json()
    assert response.json()["price_multiplier"] in [0.8, 1.0, 1.3, 1.6, 2.0]
