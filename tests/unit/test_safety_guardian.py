import numpy as np
import pytest
from src.ml.safety_guardian import SafetyGuardian

@pytest.fixture
def multipliers():
    """Provides the standard action space multipliers."""
    return [0.8, 1.0, 1.2, 1.5, 2.0]

def test_anti_gouging_rule(multipliers):
    """
    Tests RULE 1: Anti-Gouging during Disasters.
    If weather is terrible (< 0.3) and price is > 1.5x, cap it at 1.5x.
    """
    # State: Terrible weather (0.2), high traffic, normal supply
    state = np.array([0.5, 0.5, 0.5, 0.8, 0.8, 0.2])
    # Proposed action: 2.0x multiplier (index 4)
    action_idx = 4

    validated_action = SafetyGuardian.validate_action(state, action_idx, multipliers)
    # Expect the action to be capped at 1.5x
    assert multipliers[validated_action] == 1.5

def test_empty_road_protection_rule(multipliers):
    """
    Tests RULE 2: Empty Road Protection.
    If traffic is low (< 0.2), never surge above 1.2x.
    """
    # State: Good weather, low traffic (0.1), normal supply
    state = np.array([0.5, 0.5, 0.5, 0.8, 0.1, 0.9])
    # Proposed action: 1.5x multiplier (index 3)
    action_idx = 3

    validated_action = SafetyGuardian.validate_action(state, action_idx, multipliers)
    # Expect the action to be capped at 1.2x
    assert multipliers[validated_action] == 1.2

def test_supply_overshoot_rule(multipliers):
    """
    Tests RULE 3: Supply Overshoot.
    If we have huge supply (drivers > 0.8 norm), force discount or base.
    """
    # State: Normal conditions, but very high driver supply (0.9)
    state = np.array([0.5, 0.5, 0.9, 0.2, 0.5, 0.9])
    # Proposed action: 1.2x multiplier (index 2)
    action_idx = 2

    validated_action = SafetyGuardian.validate_action(state, action_idx, multipliers)
    # Expect the action to be capped at 1.0x
    assert multipliers[validated_action] == 1.0

def test_no_rule_triggered(multipliers):
    """
    Tests the case where no safety rule is triggered and the original action is returned.
    """
    # State: Normal conditions, rush hour
    state = np.array([0.7, 0.2, 0.6, 0.9, 0.8, 0.9])
    # Proposed action: 2.0x multiplier (index 4)
    action_idx = 4

    validated_action = SafetyGuardian.validate_action(state, action_idx, multipliers)
    # Expect the original action to be returned
    assert validated_action == action_idx
