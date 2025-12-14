from fastapi import APIRouter, HTTPException
from prometheus_client import Counter, Histogram, Gauge
from ml.dqn_agent import YassirPricingAgent
from .models import PredictionRequest, PredictionResponse
import os
import logging
import time
from src.utils.logging_config import PredictionLogger
from src.ml.fallback import FallbackPricingStrategy
import re
from collections import defaultdict
from config import MAX_ACTIVE_DRIVERS, MAX_PENDING_REQUESTS, MODELS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
prediction_logger = PredictionLogger()
fallback_strategy = FallbackPricingStrategy()

router = APIRouter()

# ==============================================================================
# METRICS & LOGGING
# ==============================================================================
PRICE_PREDICTIONS = Counter(
    "price_predictions_total",
    "Total number of price predictions made",
    ["zone"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Latency of price predictions",
    ["zone"]
)
PREDICTED_MULTIPLIER = Histogram(
    "predicted_multiplier",
    "Distribution of predicted price multipliers",
    ["zone"],
    buckets=[0.8, 1.0, 1.3, 1.6, 2.0]
)
SAFETY_OVERRIDES = Counter(
    "safety_override_total",
    "Number of times safety rules overrode the model",
    ["zone"]
)

# ==============================================================================
# MODEL LOADING (at startup)
# ==============================================================================
MODEL_CACHE = {}
MODEL_VERSIONS = {}
IS_HEALTHY = False

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

# ==============================================================================
# API ENDPOINTS
# ==============================================================================
@router.get("/health")
def health_check():
    if IS_HEALTHY:
        return {"status": "ok", "models_loaded": list(MODEL_CACHE.keys())}
    else:
        raise HTTPException(status_code=503, detail="Service is unhealthy, no models loaded.")

@router.get("/versions")
def get_model_versions():
    return MODEL_VERSIONS

@router.post("/predict", response_model=PredictionResponse)
def predict_price(request: PredictionRequest):
    """Predicts optimal price multiplier based on the input state"""
    start_time = time.time()

    raw_state = {
        "hour": request.hour, "day": request.day_of_week,
        "drivers": request.active_drivers, "requests": request.pending_requests,
        "traffic": request.traffic_index, "weather": request.weather_score
    }

    try:
        PRICE_PREDICTIONS.labels(zone=request.zone).inc()

        agent = MODEL_CACHE.get(request.zone)
        if not agent:
            logger.warning(f"Model not found for zone {request.zone}, using fallback")
            raw_state['day_of_week'] = raw_state.pop('day')
            multiplier = fallback_strategy.get_fallback_price(**raw_state, zone=request.zone)
            return PredictionResponse(price_multiplier=multiplier)

        with PREDICTION_LATENCY.labels(zone=request.zone).time():
            multiplier, metadata = agent.predict_price(raw_state)

        PREDICTED_MULTIPLIER.labels(zone=request.zone).observe(multiplier)
        if metadata.get('safety_override'):
            SAFETY_OVERRIDES.labels(zone=request.zone).inc()

        response_time_ms = (time.time() - start_time) * 1000
        prediction_logger.log_prediction(
            zone=request.zone,
            input_state=raw_state,
            predicted_multiplier=multiplier,
            metadata=metadata,
            response_time_ms=response_time_ms
        )

        return PredictionResponse(price_multiplier=multiplier)

    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        prediction_logger.log_error(
            zone=request.zone,
            error=str(e),
            input_state=raw_state
        )
        logger.error(f"Prediction failed for zone {request.zone}: {e}, using fallback")
        raw_state['day_of_week'] = raw_state.pop('day')
        multiplier = fallback_strategy.get_fallback_price(**raw_state, zone=request.zone)
        return PredictionResponse(price_multiplier=multiplier)
