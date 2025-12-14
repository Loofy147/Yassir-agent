import logging
import json
from datetime import datetime, timezone
from typing import Dict, Any
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class PredictionLogger:
    def __init__(self, log_file: str = "logs/predictions.jsonl"):
        self.log_file = log_file
        self.logger = logging.getLogger("predictions")
        self.logger.setLevel(logging.INFO)

        # Prevent duplicate handlers
        if not self.logger.handlers:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)

            formatter = logging.Formatter('%(message)s')
            fh.setFormatter(formatter)

            self.logger.addHandler(fh)

    def log_prediction(
        self,
        zone: str,
        input_state: Dict[str, Any],
        predicted_multiplier: float,
        metadata: Dict[str, Any],
        response_time_ms: float
    ):
        """Log a prediction with full context"""
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'zone': zone,
            'input': input_state,
            'output': {
                'multiplier': predicted_multiplier,
                'action_index': metadata.get('action'),
                'safety_override': metadata.get('safety_override', False),
                'supply_demand_ratio': metadata.get('supply_demand_ratio')
            },
            'response_time_ms': response_time_ms
        }

        self.logger.info(json.dumps(log_entry, cls=NumpyEncoder))

    def log_error(self, zone: str, error: str, input_state: Dict[str, Any]):
        """Log prediction errors"""
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': 'ERROR',
            'zone': zone,
            'error': str(error),
            'input': input_state
        }

        self.logger.error(json.dumps(log_entry, cls=NumpyEncoder))
