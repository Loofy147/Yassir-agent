import numpy as np
import pytest
from src.ml.safety_guardian import SafetyGuardian

@pytest.fixture
def multipliers():
    """Provides the standard action space multipliers."""
    return [0.8, 1.0, 1.2, 1.5, 1.8, 2.0]

def test_anti_gouging_rule(multipliers):
    """
    Tests RULE 1: Anti-Gouging during Disasters.
    If weather is terrible (< 0.3) and price is > 1.5x, cap it at 1.5x.
    """
    state = np.array([0.5, 0.5, 0.5, 0.8, 0.8, 0.2, 0.5, 1.0, 0.0, 0.0])
    action_idx = 5  # Proposed: 2.0x

    validated_action = SafetyGuardian.validate_action(state, action_idx, multipliers, 1.0, "none")
    assert multipliers[validated_action] == 1.5

def test_empty_road_protection_rule(multipliers):
    """
    Tests RULE 2: Empty Road Protection.
    If traffic is low (< 0.2), never surge above 1.2x.
    """
    state = np.array([0.5, 0.5, 0.5, 0.8, 0.1, 0.9, 0.5, 1.0, 0.0, 0.0])
    action_idx = 3  # Proposed: 1.5x

    validated_action = SafetyGuardian.validate_action(state, action_idx, multipliers, 1.0, "none")
    assert multipliers[validated_action] == 1.2

def test_supply_overshoot_rule(multipliers):
    """
    Tests RULE 3: Supply Overshoot.
    If we have huge supply (drivers > 0.8 norm), force discount or base.
    """
    state = np.array([0.5, 0.5, 0.9, 0.2, 0.5, 0.9, 0.5, 1.0, 0.0, 0.0])
    action_idx = 2  # Proposed: 1.2x

    validated_action = SafetyGuardian.validate_action(state, action_idx, multipliers, 1.0, "none")
    assert multipliers[validated_action] == 1.0

def test_no_rule_triggered(multipliers):
    """
    Tests the case where no safety rule is triggered and the original action is returned.
    """
    state = np.array([0.7, 0.2, 0.6, 0.9, 0.8, 0.9, 0.5, 1.0, 0.0, 0.0])
    action_idx = 5  # Proposed: 2.0x

    validated_action = SafetyGuardian.validate_action(state, action_idx, multipliers, 1.9, "none")
    assert validated_action == action_idx

def test_competitor_price_matching_rule(multipliers):
    """
    Tests RULE 4: Competitor Price Matching.
    If our price is > 20% higher than competitor, cap it.
    """
    state = np.array([0.5, 0.5, 0.5, 0.8, 0.5, 0.9, 0.5, 1.0, 0.0, 0.0])
    action_idx = 5  # Proposed: 2.0x
    competitor_price = 1.2

    validated_action = SafetyGuardian.validate_action(state, action_idx, multipliers, competitor_price, "none")
    assert multipliers[validated_action] == 1.2

def test_event_based_price_caps_rule(multipliers):
    """
    Tests RULE 5: Event-based Price Caps.
    Cap prices during concerts and holidays.
    """
    state = np.array([0.5, 0.5, 0.5, 0.8, 0.5, 0.9, 0.5, 0.0, 1.0, 0.0])
    action_idx = 5  # Proposed: 2.0x

    # Test concert event
    validated_action = SafetyGuardian.validate_action(state, action_idx, multipliers, 1.0, "concert")
    assert multipliers[validated_action] == 1.8

    # Test holiday event
    state = np.array([0.5, 0.5, 0.5, 0.8, 0.5, 0.9, 0.5, 0.0, 0.0, 1.0])
    validated_action = SafetyGuardian.validate_action(state, action_idx, multipliers, 1.0, "holiday")
    assert multipliers[validated_action] == 1.2
