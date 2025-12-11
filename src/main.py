from fastapi import FastAPI
from api.endpoints import router as api_router

app = FastAPI(
    title="Yassir Dynamic Pricing API",
    description="A service to provide dynamic pricing recommendations for Yassir rides.",
    version="0.1.0",
)

app.include_router(api_router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    # This is where you can add any startup logic, like loading models.
    # In our case, the model loading is handled in the endpoints file.
    pass

@app.get("/")
def read_root():
    return {"message": "Welcome to the Yassir Dynamic Pricing API"}
