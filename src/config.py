import os

# ==============================================================================
# CONFIGURATION SETTINGS
# ==============================================================================

# Action space definition
ACTION_SPACE = [0.8, 1.0, 1.2, 1.5, 2.0]

# Normalization constants
MAX_ACTIVE_DRIVERS = 200
MAX_PENDING_REQUESTS = 400

# Model directory
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
