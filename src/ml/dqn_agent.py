import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# ============================================================================
# DQN NETWORK: Maps state → Q-values for each pricing action
# ============================================================================
class PricingDQN(nn.Module):
    def __init__(self, state_dim=7, action_dim=5):  # Now 7 features (added ratio)
        super(PricingDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# ============================================================================
# REPLAY BUFFER: Stores experience tuples for off-policy learning
# ============================================================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

# ============================================================================
# PRODUCTION-READY DQN AGENT: Safety masking + Feature engineering
# ============================================================================
class YassirPricingAgent:
    def __init__(self, state_dim=7, action_dim=5, lr=0.001, zone_config=None):
        """
        Args:
            zone_config: Dict with zone-specific limits (max_drivers, max_requests)
                        Used for normalization. Example: {"max_drivers": 150, "max_requests": 300}
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Zone configuration for feature engineering
        self.zone_config = zone_config or {"max_drivers": 100, "max_requests": 200}

        # Networks
        self.policy_net = PricingDQN(state_dim, action_dim).to(self.device)
        self.target_net = PricingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=50000)

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 0.3  # START LOWER after offline pre-training
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 128
        self.target_update_freq = 1000
        self.steps = 0

        # Action to multiplier mapping
        self.multipliers = [0.8, 1.0, 1.3, 1.6, 2.0]

        # Business rule thresholds (Yassir-specific)
        self.MAX_SURGE_TRAFFIC_THRESHOLD = 0.4  # Only surge if traffic > 40%
        self.DISCOUNT_RATIO_THRESHOLD = 1.5  # Discount if drivers > 1.5x requests

    def engineer_state(self, raw_state):
        """
        IMPROVEMENT A: Feature Engineering
        Convert raw inputs to ML-friendly state with supply/demand ratio

        Args:
            raw_state: Dict with keys [hour, day, drivers, requests, traffic, weather]
        Returns:
            Normalized numpy array (7 features)
        """
        hour = raw_state["hour"] / 23.0
        day = raw_state["day"] / 6.0
        drivers_norm = raw_state["drivers"] / self.zone_config["max_drivers"]
        requests_norm = raw_state["requests"] / self.zone_config["max_requests"]
        traffic = raw_state["traffic"]  # Already 0-1
        weather = raw_state["weather"]  # Already 0-1

        # NEW FEATURE: Supply/demand ratio (solves zone size problem)
        supply_demand_ratio = raw_state["drivers"] / (raw_state["requests"] + 1e-5)
        supply_demand_ratio = min(supply_demand_ratio / 3.0, 1.0)  # Normalize to 0-1

        return np.array([hour, day, drivers_norm, requests_norm, traffic,
                        weather, supply_demand_ratio], dtype=np.float32)

    def select_action(self, state, training=True):
        """
        Selects an action using an epsilon-greedy strategy, but only from
        the set of safe actions.
        """
        mask = self.get_safe_action_mask(state)
        valid_actions = np.where(mask)[0]

        if training and random.random() < self.epsilon:
            # Exploration: choose a random valid action
            return np.random.choice(valid_actions)

        # Exploitation: choose the best valid action from the policy
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).squeeze(0)

            # Apply mask before selecting the best action
            q_values[~mask] = -float('inf')

            return q_values.argmax().item()

    def get_safe_action_mask(self, state):
        """
        Returns a boolean mask of valid actions based on safety rules.
        - True means the action is allowed.
        - False means the action is disallowed.
        """
        mask = np.array([True] * self.action_dim)

        # Extract features for rule evaluation
        traffic = state[4]
        weather = state[5]
        supply_demand_ratio = state[6]

        # RULE 1: No surge pricing if traffic is low
        if traffic < self.MAX_SURGE_TRAFFIC_THRESHOLD:
            mask[2:] = False  # Actions for 1.3x, 1.6x, 2.0x are disallowed

        # RULE 2: Must discount if there is a driver oversupply
        if supply_demand_ratio > self.DISCOUNT_RATIO_THRESHOLD:
            mask[1:] = False  # Only 0.8x is allowed

        # RULE 3: Cap surge during extreme weather
        if weather < 0.3:
            mask[4] = False  # 2.0x surge is disallowed

        return mask

    def select_safe_action(self, state):
        """
        Selects the best safe action based on Q-values and the safety mask.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).squeeze(0)

            mask = self.get_safe_action_mask(state)

            # Apply mask: set Q-values of unsafe actions to a very low number
            q_values[~mask] = -float('inf')

            return q_values.argmax().item()

    def get_multiplier(self, action):
        """Convert action index to actual price multiplier"""
        return self.multipliers[action]

    def predict_price(self, raw_state):
        """
        PRODUCTION INFERENCE METHOD

        Args:
            raw_state: Dict with keys [hour, day, drivers, requests, traffic, weather]
        Returns:
            multiplier: Float (0.8 to 2.0)
            metadata: Dict with debugging info
        """
        self.policy_net.eval()

        # Engineer features
        state = self.engineer_state(raw_state)

        # Get safe action
        action = self.select_safe_action(state)
        multiplier = self.get_multiplier(action)

        # Return with metadata for logging/debugging
        return multiplier, {
            "action": action,
            "supply_demand_ratio": state[6],
            "safety_override": action != self.select_action(state, training=False)
        }

    def store_transition(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """Perform one gradient descent step"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q_values = self.target_net(next_states)

            # Apply safety mask to next_states' Q-values
            next_state_masks = np.array([self.get_safe_action_mask(s) for s in next_states.cpu().numpy()])
            next_q_values[~torch.from_numpy(next_state_masks).to(self.device)] = -float('inf')

            next_q = next_q_values.max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def offline_pretrain(self, historical_data, epochs=50):
        """
        IMPROVEMENT C: Cold Start Prevention
        Pre-train on historical Yassir logs before live deployment

        Args:
            historical_data: List of tuples (state_dict, actual_multiplier_used, outcome_gmv, outcome_cancel_rate)
        """
        print(f"Offline pre-training on {len(historical_data)} historical samples...")

        for epoch in range(epochs):
            random.shuffle(historical_data)
            total_loss = 0

            for state_dict, actual_multiplier, gmv, cancel_rate in historical_data:
                state = self.engineer_state(state_dict)

                # Convert actual multiplier to action
                action = self.multipliers.index(actual_multiplier)

                # Compute reward from historical outcome
                reward = self._compute_reward(gmv, cancel_rate)

                # Train on this single transition (behavioral cloning)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_tensor = torch.LongTensor([action]).to(self.device)
                reward_tensor = torch.FloatTensor([reward]).to(self.device)

                q_values = self.policy_net(state_tensor)
                q_value = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze()

                loss = nn.MSELoss()(q_value, reward_tensor.squeeze())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss/len(historical_data):.4f}")

        print("Pre-training complete. Epsilon reduced to 0.3 for safer exploration.")

    def _compute_reward(self, gmv: float, completed_rides: int, lost_demand: int) -> float:
        """
        Computes a reward based on financial outcomes and market health.

        - Positive reward for GMV.
        - Penalty for lost demand (unhappy users).
        - Small bonus for each completed ride (driver satisfaction).
        """
        gmv_reward = gmv / 1000.0  # Scale down GMV
        lost_demand_penalty = (lost_demand ** 1.2) * 0.1 # Exponential penalty for many lost rides
        completion_bonus = completed_rides * 0.05

        return gmv_reward - lost_demand_penalty + completion_bonus

    def save_model(self, path="yassir_production_dqn.pth"):
        """Save model with zone config"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'zone_config': self.zone_config
        }, path)

    def load_model(self, path="yassir_production_dqn.pth"):
        """Load pre-trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.zone_config = checkpoint.get('zone_config', self.zone_config)
