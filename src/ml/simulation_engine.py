import numpy as np
import pandas as pd
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ==============================================================================
# CONFIGURATION (Business Rules)
# ==============================================================================
@dataclass
class ZoneProfile:
    name: str
    base_demand: int        # Avg requests per hour
    price_elasticity: float # How fast people quit when price goes up (Higher = more sensitive)
    traffic_factor: float   # 0.0 to 1.0 (Hydra = 0.9, suburbs = 0.3)
    driver_supply_rate: float # How easily drivers are attracted

ZONES = {
    "BAB_EZZOUAR": ZoneProfile("Bab Ezzouar", base_demand=120, price_elasticity=1.8, traffic_factor=0.5, driver_supply_rate=0.7),
    "HYDRA": ZoneProfile("Hydra", base_demand=90, price_elasticity=0.6, traffic_factor=0.95, driver_supply_rate=0.5),
    "ALGER_CENTRE": ZoneProfile("Alger Centre", base_demand=150, price_elasticity=1.1, traffic_factor=0.8, driver_supply_rate=0.8),
}

# ==============================================================================
# THE ALGIERS SIMULATOR
# ==============================================================================
class AlgiersCitySimulator:
    def __init__(self, zone_name: str, start_day: int = 0):
        if zone_name not in ZONES:
            raise ValueError(f"Unknown zone: {zone_name}")

        self.zone = ZONES[zone_name]
        self.day = start_day       # 0 = Sunday, ..., 4 = Thursday, 5 = Friday
        self.hour = 6              # Start at 6:00 AM
        self.step_count = 0

        # State Variables
        self.active_drivers = int(self.zone.base_demand * 0.8)
        self.pending_requests = 0
        self.traffic_index = 0.0
        self.weather_score = 1.0 # 1.0 = Sunny, 0.0 = Storm

        # Performance Metrics
        self.total_gmv = 0.0
        self.total_lost_rides = 0

    def get_time_factors(self):
        """Returns demand multipliers based on Algiers time patterns"""
        # Weekday: Sun(0) - Thu(4). Weekend: Fri(5) - Sat(6)
        is_weekend = self.day >= 5

        # Rush Hours: 7-9AM and 4-7PM (16-19)
        if not is_weekend:
            morning_rush = 1.5 if 7 <= self.hour <= 9 else 0.0
            evening_rush = 1.8 if 16 <= self.hour <= 19 else 0.0
            demand_time_factor = 1.0 + morning_rush + evening_rush
        else:
            # Friday: Low demand morning (prayer), high evening
            if self.day == 5 and 12 <= self.hour <= 14:
                demand_time_factor = 0.2 # Prayer dip
            elif self.day == 5 and self.hour > 18:
                demand_time_factor = 1.6 # Friday night outing
            else:
                demand_time_factor = 1.1

        return demand_time_factor

    def update_environment(self):
        """Updates traffic and weather naturally"""
        # Weather changes slowly (Markov chain)
        if random.random() < 0.05: # 5% chance weather changes
            change = np.random.normal(0, 0.2)
            self.weather_score = np.clip(self.weather_score + change, 0.2, 1.0)

        # Traffic depends on Time + Random Accidents
        time_factor = self.get_time_factors()
        base_traffic = self.zone.traffic_factor * (time_factor / 3.0) # Rush hour = traffic
        noise = np.random.normal(0, 0.05)
        self.traffic_index = np.clip(base_traffic + noise, 0.1, 1.0)

    def step(self, price_multiplier: float) -> dict:
        """
        The Core Loop:
        1. We set a price (Action).
        2. Market reacts (Outcome).
        3. Time advances (Next State).
        """
        # 1. GENERATE ORGANIC DEMAND
        time_factor = self.get_time_factors()
        # Demand = Base * TimeFactor * Weather (Rain = High Demand)
        weather_demand_boost = 1.0 + (1.0 - self.weather_score)

        raw_demand = self.zone.base_demand * time_factor * weather_demand_boost
        raw_demand = int(np.random.poisson(raw_demand))

        # 2. MARKET REACTION (ECONOMICS 101)
        # Higher Price = Lower Conversion (The "Curve")
        # Formula: Conversion = exp( -Elasticity * (Price - 1.0) )
        # If Price=1.0 -> Conv=100%. If Price=2.0 -> Conv drops fast.
        conversion_rate = math.exp(-self.zone.price_elasticity * (price_multiplier - 1.0))
        conversion_rate = np.clip(conversion_rate, 0.05, 1.0)

        actual_requests = int(raw_demand * conversion_rate)
        lost_demand_due_to_price = raw_demand - actual_requests

        # 3. MATCHING LOGIC (Supply vs Demand)
        # Traffic kills efficiency. High traffic = fewer rides completed.
        driver_efficiency = 1.0 - (self.traffic_index * 0.5)
        ride_capacity = int(self.active_drivers * driver_efficiency * 1.5) # 1.5 rides per hour approx

        completed_rides = min(actual_requests, ride_capacity)
        unfulfilled_requests = max(0, actual_requests - ride_capacity) # Drivers too busy

        # 4. FINANCIAL OUTCOME
        base_fare = 300 # Average ride price in DZD
        step_gmv = completed_rides * base_fare * price_multiplier

        # 5. DRIVER REACTION (For next step)
        # High earnings attract drivers. Low earnings push them away.
        target_drivers = int(self.active_drivers * (1.0 + (price_multiplier - 1.0) * 0.5))
        # Smooth transition (drivers don't teleport)
        self.active_drivers += int((target_drivers - self.active_drivers) * 0.2)
        self.active_drivers = max(5, self.active_drivers) # Minimum 5 drivers

        # 6. ADVANCE TIME
        self.step_count += 1
        self.hour += 1
        if self.hour >= 24:
            self.hour = 0
            self.day = (self.day + 1) % 7

        self.update_environment()

        # RETURN LOG (The "Data")
        return {
            "zone": self.zone.name,
            "day": self.day,
            "hour": self.hour,
            "active_drivers": self.active_drivers,
            "pending_requests": raw_demand, # Users opening app
            "traffic_index": round(self.traffic_index, 2),
            "weather_score": round(self.weather_score, 2),
            "price_multiplier": price_multiplier,
            "conversion_rate": round(conversion_rate, 2),
            "completed_rides": completed_rides,
            "lost_demand": lost_demand_due_to_price + unfulfilled_requests,
            "gmv": round(step_gmv, 2)
        }

# ==============================================================================
# DATA GENERATION SCRIPT
# ==============================================================================
def generate_training_dataset(days=30):
    print("🚀 Starting 'Digital Twin' Simulation for Yassir (Algiers)...")
    sim = AlgiersCitySimulator("HYDRA") # Simulating Hydra Zone

    dataset = []

    # Run simulation for 30 days
    total_steps = days * 24

    for _ in range(total_steps):
        # LOGIC: A simple "Rule Based" agent to generate history
        # (This mimics what Yassir might currently do manually)

        current_traffic = sim.traffic_index
        current_demand = sim.get_time_factors()

        # Simple heuristic rule for price
        if current_demand > 1.5 or sim.active_drivers < 10:
            action_price = 1.5 # Surge
        elif current_traffic > 0.8:
            action_price = 1.2 # Mild surge for traffic compensation
        else:
            action_price = 1.0 # Base price

        # Add some randomness (Exploration) so our AI can learn from mistakes
        if random.random() < 0.2:
            action_price = random.choice([0.8, 1.0, 1.3, 1.6, 2.0])

        # Run Step
        data_point = sim.step(action_price)
        dataset.append(data_point)

    df = pd.DataFrame(dataset)
    print(f"✅ Generated {len(df)} hours of historical data.")
    return df

# Example Run
if __name__ == "__main__":
    df = generate_training_dataset(days=7) # Generate 1 week for test
    print(df.head(10))
    # df.to_csv("yassir_hydra_simulation.csv", index=False)