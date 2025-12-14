from typing import Dict

class StaticPricingAgent:
    """
    A simple, rule-based pricing agent that serves as a baseline for comparison.
    """
    def predict_price(self, raw_state: Dict[str, any]) -> float:
        """
        Predicts a price multiplier based on a simple set of rules.
        """
        hour = raw_state.get("hour", 12)
        day = raw_state.get("day", 3)
        active_drivers = raw_state.get("drivers", 100)
        pending_requests = raw_state.get("requests", 50)

        # Rule 1: Surge pricing during rush hour
        if 7 <= hour <= 9 or 16 <= hour <= 19 and day < 5:
            return 1.3

        # Rule 2: Discount pricing if there is an oversupply of drivers
        if active_drivers > pending_requests * 2:
            return 0.8

        return 1.0
