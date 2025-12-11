from fastapi import APIRouter, HTTPException
import numpy as np
import os
import logging
from .models import PredictionRequest, PredictionResponse
from ml.ppo_agent import PPOAgent
from ml.safety_guardian import SafetyGuardian
from config import ACTION_SPACE, MAX_ACTIVE_DRIVERS, MAX_PENDING_REQUESTS, MODELS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Global state for application health and model cache
IS_HEALTHY = True
MODEL_CACHE = {}

@router.on_event("startup")
def load_models():
    """Load all models at startup."""
    global IS_HEALTHY
    try:
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)

        # Group models by zone
        models_by_zone = {}
        for model_file in os.listdir(MODELS_DIR):
            if model_file.endswith(".pth"):
                zone = model_file.split("-v")[0]
                if zone not in models_by_zone:
                    models_by_zone[zone] = []
                models_by_zone[zone].append(model_file)

        # Load the latest version of each model
        for zone, model_files in models_by_zone.items():
            latest_model = sorted(model_files, reverse=True)[0]
            model_path = os.path.join(MODELS_DIR, latest_model)
            MODEL_CACHE[zone] = PPOAgent(model_path=model_path)
            logger.info(f"Model for zone '{zone}' loaded successfully from '{latest_model}'.")

        if not MODEL_CACHE:
            IS_HEALTHY = False
            logger.warning(f"No models found in the models directory '{MODELS_DIR}'. The service will be unhealthy.")
    except Exception as e:
        IS_HEALTHY = False
        logger.critical(f"An error occurred while loading models: {e}. The service will be unhealthy.")

@router.get("/health", status_code=200)
def health_check():
    """Health check endpoint."""
    if not IS_HEALTHY:
        raise HTTPException(status_code=503, detail="Service is unhealthy: Models not loaded.")
    return {"status": "ok"}

@router.post("/predict", response_model=PredictionResponse)
def predict_price(request: PredictionRequest):
    """Predicts the optimal price multiplier."""
    if not IS_HEALTHY:
        raise HTTPException(status_code=503, detail="Service is unhealthy: Models not loaded.")

    agent = MODEL_CACHE.get(request.zone)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Model for zone '{request.zone}' not found.")

    # 1. Preprocess the input data into a normalized state vector
    state = np.array([
        request.hour / 24.0,
        request.day_of_week / 7.0,
        min(request.active_drivers / MAX_ACTIVE_DRIVERS, 1.0),
        min(request.pending_requests / MAX_PENDING_REQUESTS, 1.0),
        request.traffic_index,
        request.weather_score
    ], dtype=np.float32)

    # 2. Use the PPO agent to select an action
    action_idx, _ = agent.select_action(state, training=False)

    # 3. Apply the safety guardian
    safe_action_idx = SafetyGuardian.validate_action(state, action_idx, ACTION_SPACE)

    # 4. Get the final price multiplier from the action space
    price_multiplier = ACTION_SPACE[safe_action_idx]

    return PredictionResponse(price_multiplier=price_multiplier)

@router.get("/metrics")
def get_metrics():
    """Returns application metrics."""
    # This is a placeholder. The actual metrics logic will be added later.
    return {"metrics": "not implemented"}
