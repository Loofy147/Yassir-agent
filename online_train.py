import datetime
import os
import numpy as np
from src.ml.dqn_agent import YassirPricingAgent
from src.ml.simulation_engine import AlgiersCitySimulator

# ============================================================================
# HYPERPARAMETERS & CONFIGURATION
# ============================================================================
ZONE_NAME = "BAB_EZZOUAR"
NUM_EPISODES = 500  # An "episode" is one full day of simulation (24 hours)
MODELS_DIR = "models"
ZONE_CONFIG = {"max_drivers": 150, "max_requests": 300}

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
def run_online_training():
    """
    Trains the DQN agent by having it interact with the Algiers simulator in real-time.
    """
    print(f"--- Starting Online Reinforcement Learning for Zone: {ZONE_NAME} ---")

    # 1. Initialize the Agent and the Environment
    agent = YassirPricingAgent(zone_config=ZONE_CONFIG)
    simulator = AlgiersCitySimulator(zone_name=ZONE_NAME)

    total_steps = 0
    for episode in range(NUM_EPISODES):
        # Reset the simulator to the start of a new day
        simulator = AlgiersCitySimulator(zone_name=ZONE_NAME, start_day=episode % 7)

        # Get the initial state from the simulator
        raw_state = {
            "hour": simulator.hour, "day": simulator.day, "drivers": simulator.active_drivers,
            "requests": simulator.pending_requests, "traffic": simulator.traffic_index,
            "weather": simulator.weather_score
        }
        state = agent.engineer_state(raw_state)

        episode_reward = 0
        episode_loss = 0

        # Run the simulation for one full day (24 steps/hours)
        for hour in range(24):
            total_steps += 1

            # 2. Agent selects an action based on the current state
            action_idx = agent.select_action(state)
            price_multiplier = agent.get_multiplier(action_idx)

            # 3. Simulator executes the action and returns the outcome
            sim_result = simulator.step(price_multiplier)

            # 4. Calculate the reward
            # A good reward function is crucial. Here we balance GMV with lost demand.
            reward = (sim_result["gmv"] / 1000) - (sim_result["lost_demand"] * 0.1)

            # 5. Get the next state
            next_raw_state = {
                "hour": sim_result["hour"], "day": sim_result["day"], "drivers": sim_result["active_drivers"],
                "requests": sim_result["pending_requests"], "traffic": sim_result["traffic_index"],
                "weather": sim_result["weather_score"]
            }
            next_state = agent.engineer_state(next_raw_state)

            # 6. Store the experience in the replay buffer
            # (state, action, reward, next_state, done)
            done = (hour == 23) # "Done" is true at the end of the day/episode
            agent.store_transition(state, action_idx, reward, next_state, done)

            # 7. Train the agent on a batch of past experiences
            loss = agent.train_step()

            # Update state and track metrics
            state = next_state
            episode_reward += reward
            if loss:
                episode_loss += loss

        # Log progress at the end of each episode
        if (episode + 1) % 10 == 0:
            avg_loss = episode_loss / 24 if episode_loss > 0 else 0
            print(f"Episode {episode + 1}/{NUM_EPISODES} | Total Reward: {episode_reward:.2f} | Avg Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.3f}")

    print("\n--- Online Training Complete ---")

    # 8. Save the final trained model with versioning
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    model_filename = f"{ZONE_NAME}-v{timestamp}.pth"
    model_path = os.path.join(MODELS_DIR, model_filename)
    agent.save_model(model_path)
    print(f"✅ Trained model saved to: {model_path}")

    return agent

# ============================================================================
# VALIDATION PHASE
# ============================================================================
def validate_model(agent):
    """
    Runs the trained agent through a series of test scenarios to evaluate its strategy.
    """
    print("\n--- Running Validation on Trained Agent ---")
    simulator = AlgiersCitySimulator(zone_name=ZONE_NAME)

    scenarios = {
        "Early Morning (Low Demand)": (2, 6), # 2 AM, Saturday
        "Morning Rush Hour (High Demand)": (8, 1), # 8 AM, Monday
        "Midday Oversupply": (14, 3), # 2 PM, Wednesday
        "Evening Rush Hour (Peak Demand)": (18, 4) # 6 PM, Thursday
    }

    for name, (hour, day) in scenarios.items():
        simulator.hour = hour
        simulator.day = day
        simulator.update_environment() # Get realistic traffic/weather

        raw_state = {
            "hour": simulator.hour, "day": simulator.day, "drivers": simulator.active_drivers,
            "requests": int(simulator.zone.base_demand * simulator.get_time_factors()),
            "traffic": simulator.traffic_index, "weather": simulator.weather_score
        }

        multiplier, metadata = agent.predict_price(raw_state)

        print(f"\nScenario: {name} (Hour: {hour}, Day: {day})")
        print(f"  - State: Drivers={raw_state['drivers']}, Requests={raw_state['requests']}, Traffic={raw_state['traffic']:.2f}")
        print(f"  - Agent's Decision: Price Multiplier = {multiplier}x")
        print(f"  - Metadata: {metadata}")

if __name__ == "__main__":
    trained_agent = run_online_training()
    validate_model(trained_agent)
