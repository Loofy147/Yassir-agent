import os
import datetime
import numpy as np
from src.ml.dqn_agent import YassirPricingAgent
from src.ml.simulation_engine import AlgiersCitySimulator

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
EPISODES = 10
STEPS_PER_EPISODE = 24 * 7  # Simulate one week per episode
ZONE_NAME = "BAB_EZZOUAR"
MODELS_DIR = "models"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def compute_reward(agent: YassirPricingAgent, sim_output: dict) -> float:
    """
    Computes a reward from the simulation step output using the agent's
    internal reward function.
    """
    gmv = sim_output.get("gmv", 0)
    completed_rides = sim_output.get("completed_rides", 0)
    lost_demand = sim_output.get("lost_demand", 0)

    if agent:
        return agent._compute_reward(gmv, completed_rides, lost_demand)

    # Default reward function for static agent
    gmv_reward = gmv / 1000.0
    lost_demand_penalty = (lost_demand ** 1.2) * 0.1
    completion_bonus = completed_rides * 0.05
    return gmv_reward - lost_demand_penalty + completion_bonus

# ============================================================================
# PHASE 1: INITIALIZATION
# ============================================================================
print(f"--- Initializing Training for Zone: {ZONE_NAME} ---")
env = AlgiersCitySimulator(zone_name=ZONE_NAME)
agent = YassirPricingAgent(zone_config={"max_drivers": 150, "max_requests": 300})

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# ============================================================================
# PHASE 2: VALIDATION FRAMEWORK
# ============================================================================
from src.ml.static_agent import StaticPricingAgent

def validate_agent(agent, validation_steps=100):
    """
    Runs the agent in a validation environment to assess its performance.
    """
    validation_env = AlgiersCitySimulator(zone_name=ZONE_NAME)
    total_reward = 0

    for _ in range(validation_steps):
        raw_state = {
            "hour": validation_env.hour, "day": validation_env.day,
            "drivers": validation_env.active_drivers, "requests": validation_env.pending_requests,
            "traffic": validation_env.traffic_index, "weather": validation_env.weather_score
        }

        if isinstance(agent, YassirPricingAgent):
            state = agent.engineer_state(raw_state)
            action = agent.select_action(state, training=False)
            multiplier = agent.get_multiplier(action)
        else: # Static Agent
            multiplier = agent.predict_price(raw_state)

        sim_output = validation_env.step(multiplier)
        reward = compute_reward(agent if isinstance(agent, YassirPricingAgent) else None, sim_output)
        total_reward += reward

    return total_reward / validation_steps

# ============================================================================
# PHASE 3: ONLINE RL TRAINING LOOP
# ============================================================================
print("--- Starting Online Reinforcement Learning ---")
static_agent = StaticPricingAgent()

for episode in range(EPISODES):
    # Reset environment at the start of each episode
    env = AlgiersCitySimulator(zone_name=ZONE_NAME)
    total_reward = 0
    total_loss = 0

    # Get initial state
    raw_state = {
        "hour": env.hour, "day": env.day, "drivers": env.active_drivers,
        "requests": env.pending_requests, "traffic": env.traffic_index,
        "weather": env.weather_score
    }
    state = agent.engineer_state(raw_state)

    for step in range(STEPS_PER_EPISODE):
        # 1. Agent selects an action
        action_index = agent.select_action(state, training=True)
        price_multiplier = agent.get_multiplier(action_index)

        # 2. Environment executes the action
        sim_output = env.step(price_multiplier)
        reward = compute_reward(agent, sim_output)

        # 3. Observe the next state
        next_raw_state = {
            "hour": env.hour, "day": env.day, "drivers": env.active_drivers,
            "requests": env.pending_requests, "traffic": env.traffic_index,
            "weather": env.weather_score
        }
        next_state = agent.engineer_state(next_raw_state)

        done = (step == STEPS_PER_EPISODE - 1)

        # 4. Store the transition in the replay buffer
        agent.store_transition(state, action_index, reward, next_state, done)

        # 5. Train the agent
        loss = agent.train_step()
        if loss:
            total_loss += loss

        state = next_state
        total_reward += reward

    # --- Validation Phase ---
    if (episode + 1) % 2 == 0:
        dqn_avg_reward = validate_agent(agent)
        static_avg_reward = validate_agent(static_agent)
        print(f"--- Validation after Episode {episode + 1} ---")
        print(f"  DQN Agent Avg Reward: {dqn_avg_reward:.2f}")
        print(f"  Static Agent Avg Reward: {static_avg_reward:.2f}")
        print("-" * 20)

    avg_loss = total_loss / STEPS_PER_EPISODE if STEPS_PER_EPISODE > 0 else 0
    print(f"Episode {episode + 1}/{EPISODES} | Total Reward: {total_reward:.2f} | Avg Loss: {avg_loss:.4f}")

# ============================================================================
# PHASE 4: SAVING THE MODEL
# ============================================================================
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
model_filename = f"{ZONE_NAME}-v{timestamp}.pth"
model_path = os.path.join(MODELS_DIR, model_filename)
agent.save_model(model_path)
print(f"\n✅ Training complete. Model for zone '{ZONE_NAME}' saved to {model_path}")
