import numpy as np

class SafetyGuardian:
    """
    The hard-coded business rules that OVERRIDE the AI.
    This protects the brand from 'AI hallucinations'.
    """
    @staticmethod
    def validate_action(state, action_idx, multipliers, competitor_price, event):
        """
        Input: State vector, AI proposed action index, competitor price, event
        Output: Safe action index
        """
        drivers_norm = state[2]
        traffic = state[4]
        weather = state[5]

        proposed_multiplier = multipliers[action_idx]

        # RULE 1: Anti-Gouging during Disasters
        if weather < 0.3 and proposed_multiplier > 1.5:
            return multipliers.index(1.5)

        # RULE 2: Empty Road Protection
        if traffic < 0.2 and proposed_multiplier > 1.2:
            return multipliers.index(1.2)

        # RULE 3: Supply Overshoot
        if drivers_norm > 0.8 and proposed_multiplier > 1.0:
            return multipliers.index(1.0)

        # RULE 4: Event-based Price Caps
        if event == "concert" and proposed_multiplier > 1.8:
            return multipliers.index(1.8)
        elif event == "holiday" and proposed_multiplier > 1.2:
            return multipliers.index(1.2)

        # RULE 5: Competitor Price Matching
        if proposed_multiplier > competitor_price * 1.2:
            # Find the closest multiplier that is less than or equal to competitor_price * 1.2
            for i in range(len(multipliers) - 1, -1, -1):
                if multipliers[i] <= competitor_price * 1.2:
                    return i
            return 0 # If all multipliers are too high, return the lowest

        return action_idx
