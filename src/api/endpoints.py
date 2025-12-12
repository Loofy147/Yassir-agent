from fastapi import APIRouter, HTTPException
import os
import logging
from .models import PredictionRequest, PredictionResponse
from ml.dqn_agent import YassirPricingAgent
from config import MAX_ACTIVE_DRIVERS, MAX_PENDING_REQUESTS, MODELS_DIR

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
            logger.warning(f"Models directory '{MODELS_DIR}' not found. Created it. Run train.py to populate it.")

        zone_config = {
            "max_drivers": MAX_ACTIVE_DRIVERS,
            "max_requests": MAX_PENDING_REQUESTS
        }

        for model_file in os.listdir(MODELS_DIR):
            if model_file.endswith(".pth"):
                zone = model_file.replace(".pth", "")
                model_path = os.path.join(MODELS_DIR, model_file)

                agent = YassirPricingAgent(zone_config=zone_config)
                agent.load_model(model_path)

                MODEL_CACHE[zone] = agent
                logger.info(f"Model for zone '{zone}' loaded successfully.")

        if not MODEL_CACHE:
            IS_HEALTHY = False
            logger.warning(f"No models found in '{MODELS_DIR}'. Run train.py first! Service is unhealthy.")
        else:
            IS_HEALTHY = True
            logger.info("All models loaded. Service is healthy.")

    except Exception as e:
        IS_HEALTHY = False
        logger.critical(f"Critical error loading models: {e}. The service will be unhealthy.")

@router.get("/health", status_code=200)
def health_check():
    """Health check endpoint."""
    if not IS_HEALTHY:
        raise HTTPException(status_code=503, detail="Service is unhealthy: Models not loaded.")
    return {"status": "ok"}

@router.post("/predict", response_model=PredictionResponse)
def predict_price(request: PredictionRequest):
    """Predicts the optimal price multiplier using the DQN agent."""
    if not IS_HEALTHY:
        raise HTTPException(status_code=503, detail="Service is unhealthy: Models not loaded.")

    agent = MODEL_CACHE.get(request.zone)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Model for zone '{request.zone}' not found.")

    raw_state = {
        "hour": request.hour,
        "day": request.day_of_week,
        "drivers": request.active_drivers,
        "requests": request.pending_requests,
        "traffic": request.traffic_index,
        "weather": request.weather_score
    }

    multiplier, metadata = agent.predict_price(raw_state)

    logger.info(f"Prediction for zone {request.zone}: multiplier={multiplier}, metadata={metadata}")

    return PredictionResponse(price_multiplier=multiplier)

@router.get("/metrics")
def get_metrics():
    """Returns application metrics."""
    return {"metrics": "not implemented"}
