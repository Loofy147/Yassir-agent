import pandas as pd
import numpy as np

def generate_yassir_pricing_data(num_samples=10000):
    """
    Generates synthetic training data for the Yassir Dynamic Pricing Agent.
    Simulates: Algiers rush hours, rain effects, and driver supply dynamics.
    """

    # 1. Create Time Features (Cyclical)
    # ---------------------------------------------------------
    timestamps = pd.date_range(start="2024-01-01", periods=num_samples, freq="h")
    hours = timestamps.hour.values
    days = timestamps.dayofweek.values

    # 2. Simulate Demand (Requests)
    # ---------------------------------------------------------
    # Demand peaks at 8AM (school/work) and 6PM (return home)
    base_demand = 50 + 30 * np.sin((hours - 6) * np.pi / 12)**2
    # Add randomness (events, holidays)
    requests = np.random.normal(base_demand, 10).astype(int)
    requests = np.maximum(5, requests) # Minimum 5 requests

    # 3. Simulate Supply (Active Drivers)
    # ---------------------------------------------------------
    # Drivers usually lag behind demand.
    # If demand is high, drivers enter the zone, but not instantly.
    supply_ratio = np.random.uniform(0.7, 1.2, size=num_samples) # 70% to 120% of demand
    drivers = (requests * supply_ratio).astype(int)

    # 4. Context Features
    # ---------------------------------------------------------
    # Traffic: High during rush hours (8am-9am, 5pm-7pm)
    traffic_noise = np.random.normal(0, 0.1, size=num_samples)
    traffic_index = 0.3 + 0.5 * np.sin((hours - 7) * np.pi / 12)**2 + traffic_noise
    traffic_index = np.clip(traffic_index, 0.1, 0.95)

    # Weather: 0.0 (Storm) to 1.0 (Sunny). Algiers is mostly sunny.
    weather_score = np.random.beta(5, 2, size=num_samples)

    # 5. Simulate Historical Human Actions & Outcomes
    # ---------------------------------------------------------
    # Randomly assign actions that human operators might have taken
    historical_prices = np.random.choice([0.8, 1.0, 1.3, 1.6, 2.0], size=num_samples)

    # Calculate Resulting Cancellation Rate (The "Environment" Logic)
    # High Price + High Traffic = High Cancellations (Frustrated users)
    # Low Price = Low Cancellations
    base_cancel = 0.02
    price_sensitivity = 0.08 * (historical_prices - 1.0) # Users hate surges
    traffic_friction = 0.1 * traffic_index # Traffic makes users cancel

    cancellation_rate = base_cancel + price_sensitivity + traffic_friction
    cancellation_rate = np.clip(cancellation_rate, 0.0, 1.0)

    # Calculate GMV (Gross Merchandise Volume)
    avg_fare_dzd = 300
    gmv = requests * (1 - cancellation_rate) * avg_fare_dzd * historical_prices

    # 6. Create DataFrame
    df = pd.DataFrame({
        'hour': hours,                # State[0]
        'day': days,                  # State[1]
        'drivers': drivers,           # State[2]
        'requests': requests,         # State[3]
        'traffic': traffic_index,     # State[4]
        'weather': weather_score,     # State[5]
        'action_multiplier': historical_prices,
        'reward_gmv': gmv,
        'reward_cancel_rate': cancellation_rate
    })

    return df