# Yassir Dynamic Pricing Service

This project is a multi-tenant, AI-powered dynamic pricing service for the Yassir ride-sharing platform. It uses a Deep Q-Network (DQN) Reinforcement Learning agent to learn optimal pricing strategies from a simulated market environment. The trained agent is then exposed via a FastAPI service for real-time predictions.

## Project Structure

```
.
├── Dockerfile
├── README.md
├── models/
│   └── HYDRA.pth
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── endpoints.py
│   │   └── models.py
│   ├── config.py
│   ├── main.py
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── ppo_agent.py
│   │   ├── safety_guardian.py
│   │   └── simulation_engine.py
│   └── train.py
└── tests/
    └── unit/
        └── test_safety_guardian.py
```

-   **`models/`**: This directory stores the trained model files for each zone.
-   **`src/`**: This directory contains the main application code.
    -   **`api/`**: This directory contains the FastAPI application.
    -   **`ml/`**: This directory contains the machine learning code, including the PPO agent, safety guardian, and simulation engine.
    -   **`config.py`**: This file contains the application's configuration.
    -   **`main.py`**: This is the entry point for the FastAPI application.
    -   **`train.py`**: This script is used to train the PPO agent.
-   **`tests/`**: This directory contains the unit and integration tests.

## Setup Instructions

1.  **Clone the repository:**
    ```
    git clone <repository-url>
    ```
2.  **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

## How to Run

### Training

To train a new model, run the `train.py` script:

```
python src/train.py
```

This will create a `models/HYDRA-v<timestamp>.pth` file. The API will automatically load the latest version of the model for each zone.

### Running the API Server

To run the API server, use the following command:

```
PYTHONPATH=src uvicorn src.main:app --host 0.0.0.0 --port 8000
```

To use a custom model directory, set the `MODELS_DIR` environment variable:

```
PYTHONPATH=src MODELS_DIR=custom_models uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### Running with Docker

To build and run the Docker container, use the following commands:

```
docker build -t yassir-pricing-service .
docker run -p 8000:8000 yassir-pricing-service
```

## API Usage

### Health Check

To check the health of the service, send a GET request to the `/api/health` endpoint:

```
curl http://localhost:8000/api/health
```

### Predict Price

To predict the price for a given zone, send a POST request to the `/api/predict` endpoint:

```
curl -X POST -H "Content-Type: application/json" -d '{"zone": "HYDRA", "hour": 10, "day_of_week": 2, "active_drivers": 100, "pending_requests": 50, "traffic_index": 0.5, "weather_score": 0.8, "event": "concert", "competitor_price": 1.1}' http://localhost:8000/api/predict
```

## Testing

To run the unit tests, use the following command:

```
PYTHONPATH=. pytest
```
