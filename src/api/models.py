from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    """
    Defines the structure for a price prediction request.
    The data sent to the /predict endpoint must conform to this schema.
    """
    zone: str = Field(..., description="The zone for which to predict the price.")
    hour: int = Field(..., ge=0, le=23, description="The hour of the day (0-23).")
    day_of_week: int = Field(..., ge=0, le=6, description="The day of the week (0=Sunday, 6=Saturday).")
    active_drivers: int = Field(..., ge=0, description="The number of active drivers in the zone.")
    pending_requests: int = Field(..., ge=0, description="The number of pending ride requests.")
    traffic_index: float = Field(..., ge=0.0, le=1.0, description="A normalized traffic index (0=clear, 1=congested).")
    weather_score: float = Field(..., ge=0.0, le=1.0, description="A normalized weather score (0=storm, 1=sunny).")

class PredictionResponse(BaseModel):
    """
    Defines the structure for a price prediction response.
    The data returned from the /predict endpoint will conform to this schema.
    """
    price_multiplier: float = Field(..., description="The recommended price multiplier.")
