from fastapi import FastAPI
from api.endpoints import router as api_router
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="Yassir Dynamic Pricing API",
    description="An API for predicting dynamic price multipliers.",
    version="1.0.0"
)

# Instrument the app with Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Include the API router
app.include_router(api_router, prefix="/api")

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Yassir Dynamic Pricing API"}
