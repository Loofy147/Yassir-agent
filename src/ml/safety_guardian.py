import numpy as np

class SafetyGuardian:
    """
    The hard-coded business rules that OVERRIDE the AI.
    This protects the brand from 'AI hallucinations'.
    """
    @staticmethod
    def validate_action(state, action_idx, multipliers):
        """
        Input: State vector, AI proposed action index
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

        return action_idx
