from typing import Dict

class FallbackPricingStrategy:
    """Simple rule-based pricing when ML model fails"""

    @staticmethod
    def get_fallback_price(
        zone: str,
        hour: int,
        day_of_week: int,
        active_drivers: int,
        pending_requests: int,
        traffic_index: float,
        weather_score: float
    ) -> float:
        """
        Simple heuristic pricing strategy.

        Rules:
        1. Surge during rush hour (7-9am, 4-7pm on weekdays)
        2. Discount when supply > demand
        3. Weather adjustment
        4. Traffic adjustment
        """
        multiplier = 1.0  # Start with base price

        # Rush hour surge (weekdays only)
        is_weekday = day_of_week < 5
        is_rush_hour = (7 <= hour <= 9) or (16 <= hour <= 19)
        if is_weekday and is_rush_hour:
            multiplier = 1.3

        # Supply-demand adjustment
        if pending_requests > 0:
            supply_demand_ratio = active_drivers / pending_requests
            if supply_demand_ratio > 1.5:
                # Too many drivers, discount
                multiplier = min(multiplier, 0.8)
            elif supply_demand_ratio < 0.5:
                # Not enough drivers, surge
                multiplier = max(multiplier, 1.6)

        # Weather adjustment (bad weather = more demand)
        if weather_score < 0.3:
            multiplier *= 1.2

        # Traffic adjustment
        if traffic_index > 0.7:
            multiplier *= 1.1

        # Cap at reasonable bounds
        multiplier = max(0.8, min(2.0, multiplier))

        # Round to nearest valid action
        valid_multipliers = [0.8, 1.0, 1.3, 1.6, 2.0]
        multiplier = min(valid_multipliers, key=lambda x: abs(x - multiplier))

        return multiplier
