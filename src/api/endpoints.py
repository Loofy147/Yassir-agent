from fastapi import APIRouter, HTTPException
import os
import logging
import re
from collections import defaultdict
from .models import PredictionRequest, PredictionResponse
from ml.dqn_agent import YassirPricingAgent
from config import MAX_ACTIVE_DRIVERS, MAX_PENDING_REQUESTS, MODELS_DIR
from prometheus_client import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------- Prometheus Custom Metrics ----------------
# A counter to track the number of price predictions per zone.
PRICE_PREDICTIONS = Counter(
    "price_predictions_total",
    "Total number of price predictions made",
    ["zone"] # Label to distinguish predictions by zone
)
# -----------------------------------------------------------

IS_HEALTHY = True
MODEL_CACHE = {}
MODEL_VERSIONS = {}

@router.on_event("startup")
def load_models():
    """Load the latest version of all models at startup."""
    global IS_HEALTHY, MODEL_CACHE, MODEL_VERSIONS
    MODEL_CACHE = {}
    MODEL_VERSIONS = {}
    try:
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
            logger.warning(f"Models directory '{MODELS_DIR}' not found. Created it.")

        zone_config = { "max_drivers": MAX_ACTIVE_DRIVERS, "max_requests": MAX_PENDING_REQUESTS }

        model_files = defaultdict(list)
        version_pattern = re.compile(r"(.+)-v(\d{14})\.pth")

        for filename in os.listdir(MODELS_DIR):
            match = version_pattern.match(filename)
            if match:
                zone, _ = match.groups()
                model_files[zone].append(filename)

        for zone, files in model_files.items():
            files.sort(reverse=True)
            latest_model_file = files[0]
            model_path = os.path.join(MODELS_DIR, latest_model_file)

            agent = YassirPricingAgent(zone_config=zone_config)
            agent.load_model(model_path)

            MODEL_CACHE[zone] = agent
            MODEL_VERSIONS[zone] = latest_model_file
            logger.info(f"Successfully loaded model for zone '{zone}': {latest_model_file}")

        if not MODEL_CACHE:
            IS_HEALTHY = False
            logger.warning(f"No valid models found in '{MODELS_DIR}'. Service is unhealthy.")
        else:
            IS_HEALTHY = True
            logger.info("All models loaded. Service is healthy.")

    except Exception as e:
        IS_HEALTHY = False
        logger.critical(f"Critical error during model loading: {e}", exc_info=True)

@router.get("/health", status_code=200)
def health_check():
    """Health check endpoint."""
    if not IS_HEALTHY:
        raise HTTPException(status_code=503, detail="Service is unhealthy: Models not loaded.")
    return {"status": "ok", "loaded_models": list(MODEL_CACHE.keys())}

@router.get("/versions", status_code=200)
def get_model_versions():
    """Returns the filenames of the currently loaded models."""
    if not IS_HEALTHY:
        raise HTTPException(status_code=503, detail="Service is unhealthy.")
    return MODEL_VERSIONS

@router.post("/predict", response_model=PredictionResponse)
def predict_price(request: PredictionRequest):
    """Predicts the optimal price multiplier using the latest DQN agent."""
    # Increment the custom Prometheus counter for this zone.
    PRICE_PREDICTIONS.labels(zone=request.zone).inc()

    if not IS_HEALTHY:
        raise HTTPException(status_code=503, detail="Service is unhealthy.")

    agent = MODEL_CACHE.get(request.zone)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Model for zone '{request.zone}' not found.")

    raw_state = {
        "hour": request.hour, "day": request.day_of_week, "drivers": request.active_drivers,
        "requests": request.pending_requests, "traffic": request.traffic_index, "weather": request.weather_score
    }

    multiplier, metadata = agent.predict_price(raw_state)

    model_version = MODEL_VERSIONS.get(request.zone, "unknown")
    logger.info(f"Prediction for zone {request.zone} (model: {model_version}): multiplier={multiplier}")

    return PredictionResponse(price_multiplier=multiplier)
