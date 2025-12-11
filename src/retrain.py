import torch
import numpy as np
import math
import argparse
import os
from datetime import datetime

from ml.simulation_engine import AlgiersCitySimulator
from ml.ppo_agent import PPOAgent
from config import ACTION_SPACE, MAX_ACTIVE_DRIVERS, MAX_PENDING_REQUESTS
from train import YassirPricingEnv

def retrain_model(model_path: str, zone: str, episodes: int = 100):
    """
    Loads an existing model, generates new data, and retrains the model.

    Args:
        model_path (str): The path to the model to be retrained.
        zone (str): The zone for which to retrain the model.
        episodes (int, optional): The number of episodes to retrain for. Defaults to 100.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # 1. SETUP
    sim = AlgiersCitySimulator(zone)
    env = YassirPricingEnv(sim)
    agent = PPOAgent(state_dim=env.state_dim, action_dim=len(env.action_space), lr=0.0001, model_path=model_path)

    print(f"🚀 Starting retraining for zone '{zone}' from model '{model_path}'...")

    # Hyperparameters
    max_steps = 168
    update_timestep = 24

    # 2. RETRAINING LOOP
    for episode in range(1, episodes + 1):
        state = env.reset()
        episode_reward = 0

        for t in range(max_steps):
            action_idx, action_prob = agent.select_action(state, training=True)
            next_state, reward, done, stats = env.step(action_idx)
            log_prob = math.log(action_prob + 1e-10)
            agent.put_data((state, action_idx, reward, next_state, log_prob, done))
            state = next_state
            episode_reward += reward

            if (t + 1) % update_timestep == 0:
                agent.update()

        if episode % 20 == 0:
            print(f"Episode {episode:03d} | Avg Reward: {episode_reward / max_steps:6.2f}")

    # 3. SAVE THE NEW MODEL
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    new_model_path = f"models/{zone}-v{timestamp}.pth"
    torch.save(agent.policy.state_dict(), new_model_path)
    print(f"\n✅ Retraining Complete. New model saved to '{new_model_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain a PPO agent for a specific zone.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model to be retrained.")
    parser.add_argument("--zone", type=str, required=True, help="The zone for which to retrain the model.")
    parser.add_argument("--episodes", type=int, default=100, help="The number of episodes to retrain for.")
    args = parser.parse_args()

    retrain_model(args.model_path, args.zone, args.episodes)
