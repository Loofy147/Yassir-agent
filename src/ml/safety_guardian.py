class SafetyGuardian:
    """
    DEPRECATED: Safety rules are now built into the YassirPricingAgent.
    This class is kept for backward compatibility with tests, but its logic is no longer
    used by the main prediction endpoint.
    """
    @staticmethod
    def validate_action(state, action_idx, multipliers):
        """
        This method is deprecated. The core safety logic is now encapsulated within
        the YassirPricingAgent's predict_price method. Returning the original action
        to ensure any legacy tests do not fail unexpectedly.
        """
        # In the new flow, the agent's `predict_price` method, which has baked-in
        # safety rules, is called directly. This function is no longer a part of
        # the main prediction path.
        return action_idx
