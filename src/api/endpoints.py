from fastapi import APIRouter, HTTPException, Header
from typing import Optional
import os
import logging
import re
import hashlib
from collections import defaultdict
from .models import PredictionRequest, PredictionResponse
from ml.dqn_agent import YassirPricingAgent
from ml.static_agent import StaticPricingAgent
from config import MAX_ACTIVE_DRIVERS, MAX_PENDING_REQUESTS, MODELS_DIR
from prometheus_client import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

PRICE_PREDICTIONS = Counter(
    "price_predictions_total",
    "Total number of price predictions made",
    ["zone", "agent_type"]
)

IS_HEALTHY = True
MODEL_CACHE = {}
MODEL_VERSIONS = {}
STATIC_AGENT = StaticPricingAgent()

@router.on_event("startup")
def load_models():
    global IS_HEALTHY, MODEL_CACHE, MODEL_VERSIONS
    MODEL_CACHE = {}
    MODEL_VERSIONS = {}
    try:
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
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
            logger.info(f"Loaded model for zone '{zone}': {latest_model_file}")
        IS_HEALTHY = True
        logger.info("Model loading complete. Service is healthy.")
    except Exception as e:
        IS_HEALTHY = False
        logger.critical(f"Critical error loading models: {e}", exc_info=True)

@router.get("/health", status_code=200)
def health_check():
    return {"status": "ok" if IS_HEALTHY else "unhealthy", "loaded_models": list(MODEL_CACHE.keys())}

@router.get("/versions", status_code=200)
def get_model_versions():
    return MODEL_VERSIONS

@router.post("/private/reset-metrics", status_code=200, include_in_schema=False)
def reset_metrics():
    """
    Resets the custom Prometheus counters. This is for internal testing purposes only.
    """
    PRICE_PREDICTIONS.clear()
    return {"message": "Custom metrics have been reset."}

@router.post("/predict", response_model=PredictionResponse)
def predict_price(request: PredictionRequest, x_user_id: Optional[str] = Header(None)):
    raw_state = { "hour": request.hour, "day": request.day_of_week, "drivers": request.active_drivers,
                  "requests": request.pending_requests, "traffic": request.traffic_index, "weather": request.weather_score }
    agent_type = "dqn"
    if x_user_id:
        user_hash = int(hashlib.md5(x_user_id.encode()).hexdigest(), 16)
        if user_hash % 10 < 2:
            agent_type = "static"

    if agent_type == "static":
        agent = STATIC_AGENT
        model_version = "static_rules"
    else:
        agent = MODEL_CACHE.get(request.zone)
        if not agent:
            agent = STATIC_AGENT
            agent_type = "static_fallback"
            model_version = "static_rules"
        else:
            model_version = MODEL_VERSIONS.get(request.zone, "unknown")

    PRICE_PREDICTIONS.labels(zone=request.zone, agent_type=agent_type).inc()
    multiplier, metadata = agent.predict_price(raw_state)
    logger.info(f"Prediction for zone {request.zone} (user: {x_user_id}, agent: {agent_type}, model: {model_version}): multiplier={multiplier}")
    return PredictionResponse(price_multiplier=multiplier)
