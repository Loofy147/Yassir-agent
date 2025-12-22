import pytest
from src.ml.dqn_agent import YassirPricingAgent

def test_yassir_pricing_agent_initialization():
    """
    Tests that the YassirPricingAgent initializes correctly.
    """
    agent = YassirPricingAgent()
    assert agent is not None
