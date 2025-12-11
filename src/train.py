import torch
import numpy as np
import math

from ml.simulation_engine import AlgiersCitySimulator
from ml.ppo_agent import PPOAgent
from config import ACTION_SPACE, MAX_ACTIVE_DRIVERS, MAX_PENDING_REQUESTS

# ==============================================================================
# ENVIRONMENT WRAPPER (Gym-like)
# ==============================================================================
class YassirPricingEnv:
    def __init__(self, simulator):
        self.sim = simulator
        self.action_space = ACTION_SPACE
        self.state_dim = 6

    def reset(self):
        # For simplicity in this standalone script, we re-initialize the simulator.
        # In a real-world scenario, you might have more complex reset logic.
        self.sim = AlgiersCitySimulator("HYDRA")
        return self._get_state()

    def _get_state(self):
        # Normalize inputs for the Neural Network (0.0 to 1.0 range)
        s = self.sim
        return np.array([
            s.hour / 24.0,
            s.day / 7.0,
            min(s.active_drivers / MAX_ACTIVE_DRIVERS, 1),
            min(s.pending_requests / MAX_PENDING_REQUESTS, 1),
            s.traffic_index,
            s.weather_score
        ], dtype=np.float32)

    def step(self, action_idx):
        multiplier = self.action_space[action_idx]
        stats = self.sim.step(multiplier)
        revenue_score = stats['gmv'] / 100.0
        lost_customer_penalty = stats['lost_demand'] * 0.5
        reward = revenue_score - lost_customer_penalty
        next_state = self._get_state()
        done = False # This is a continuous environment
        return next_state, reward, done, stats

# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================
if __name__ == "__main__":
    # 1. SETUP
    sim = AlgiersCitySimulator("HYDRA")
    env = YassirPricingEnv(sim)
    agent = PPOAgent(state_dim=6, action_dim=len(env.action_space), lr=0.0003)

    print("🚀 Starting PPO Training for Yassir Dynamic Pricing...")
    print(f"Zone: {sim.zone.name} | Base Demand: {sim.zone.base_demand}/hr")

    # Hyperparameters
    total_episodes = 500
    max_steps = 168
    update_timestep = 24

    # Tracking
    reward_history = []

    # 2. TRAINING LOOPS
    for episode in range(1, total_episodes + 1):
        state = env.reset()
        episode_reward = 0

        for t in range(max_steps):
            # A. Select Action
            action_idx, action_prob = agent.select_action(state, training=True)

            # B. Apply to Simulation
            next_state, reward, done, stats = env.step(action_idx)

            # Store data for PPO update
            log_prob = math.log(action_prob + 1e-10)
            agent.put_data((state, action_idx, reward, next_state, log_prob, done))

            state = next_state
            episode_reward += reward

            # C. Update Policy
            if (t + 1) % update_timestep == 0:
                agent.update()

        # Track history
        reward_history.append(episode_reward)
        avg_reward = np.mean(reward_history[-10:])

        # D. Business Logging
        if episode % 20 == 0:
            print(f"Episode {episode:03d} | Avg Reward: {avg_reward:6.2f} | "
                  f"Last GMV: {stats['gmv']:6.1f} DZD | "
                  f"Traffic: {stats['traffic_index']:.2f} | "
                  f"Price: {stats['price_multiplier']}x")

    # 3. SAVE THE TRAINED BRAIN
    print("\n✅ Training Complete. Saving Model...")
    model_path = f"models/{sim.zone.name}.pth"
    torch.save(agent.policy.state_dict(), model_path)
    print(f"Model saved to '{model_path}'")