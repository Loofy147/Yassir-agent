class StaticPricingAgent:
    """
    A simple, rule-based pricing agent that serves as a baseline for A/B testing.
    This agent's logic is deterministic and easy to understand.
    """
    def predict_price(self, raw_state: dict):
        """
        Calculates a price multiplier based on a fixed set of rules.

        Args:
            raw_state: A dictionary containing the current state, including 'hour'.

        Returns:
            A tuple containing the price multiplier and a metadata dictionary.
        """
        hour = raw_state.get("hour", 0)

        # Rule 1: Morning and evening rush hour surge
        if (7 <= hour <= 9) or (17 <= hour <= 19):
            multiplier = 1.3
            reason = "rush_hour_surge"
        # Rule 2: Late night discount
        elif 0 <= hour <= 5:
            multiplier = 0.8
            reason = "late_night_discount"
        # Default: Base price
        else:
            multiplier = 1.0
            reason = "base_price"

        metadata = {
            "agent_type": "static",
            "reason": reason
        }

        return multiplier, metadata
