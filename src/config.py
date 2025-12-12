import os

# ==============================================================================
# CONFIGURATION SETTINGS
# ==============================================================================

# Action space definition - Standardized to match the DQN agent
ACTION_SPACE = [0.8, 1.0, 1.3, 1.6, 2.0]

# Normalization constants - Standardized to match the DQN agent's zone_config
MAX_ACTIVE_DRIVERS = 150
MAX_PENDING_REQUESTS = 300

# Model directory
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
