import numpy as np
from config import MAX_ACTIVE_DRIVERS, MAX_PENDING_REQUESTS

def get_state_vector(hour, day_of_week, active_drivers, pending_requests, traffic_index, weather_score, competitor_price, event):
    """
    Creates the state vector for the PPO agent.
    """
    event_vector = [0.0, 0.0, 0.0]
    if event == "concert":
        event_vector[0] = 1.0
    elif event == "holiday":
        event_vector[1] = 1.0
    else:
        event_vector[2] = 1.0

    state = [
        hour / 24.0,
        day_of_week / 7.0,
        min(active_drivers / MAX_ACTIVE_DRIVERS, 1),
        min(pending_requests / MAX_PENDING_REQUESTS, 1),
        traffic_index,
        weather_score,
        competitor_price / 2.0
    ]
    state.extend(event_vector)

    return np.array(state, dtype=np.float32)
