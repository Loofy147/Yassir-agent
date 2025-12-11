import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque

# ==============================================================================
# 3. PPO AGENT (ACTOR-CRITIC)
# ==============================================================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Shared feature extractor
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        # Actor: Decides probability of each price
        self.actor = nn.Linear(64, action_dim)

        # Critic: Estimates value of state (How good is the current situation?)
        self.critic = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))

        # Actor output (probabilities)
        action_probs = F.softmax(self.actor(x), dim=-1)

        # Critic output (value)
        state_value = self.critic(x)

        return action_probs, state_value

class PPOAgent:
    def __init__(self, state_dim=6, action_dim=5, lr=0.002, model_path=None):
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 4

        self.policy = ActorCritic(state_dim, action_dim)
        if model_path:
            self.policy.load_state_dict(torch.load(model_path))
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.data = [] # Store trajectory

    def put_data(self, item):
        self.data.append(item)

    def select_action(self, state, training=True):
        state_t = torch.FloatTensor(state)
        with torch.no_grad():
            action_probs, _ = self.policy_old(state_t)

        # Sample action based on probability
        dist = Categorical(action_probs)
        action = dist.sample()

        # Apply SAFETY GUARDIAN during inference (and training if desired)
        # We let the agent explore, but in production, we clamp here.
        # For training, we return the raw action to learn "real" consequences,
        # OR we clamp it to teach it legal moves.
        # Strategy: Train on Raw, Deploy with Guardian.

        return action.item(), action_probs[action].item()

    def update(self):
        # Unpack data
        states, actions, rewards, next_states, old_log_probs, dones = zip(*self.data)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        old_log_probs = torch.FloatTensor(np.array(old_log_probs))

        # Monte Carlo estimate of returns
        rewards_normalized = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_normalized.insert(0, discounted_reward)

        rewards_normalized = torch.FloatTensor(rewards_normalized)
        # Normalize rewards for stability
        rewards_normalized = (rewards_normalized - rewards_normalized.mean()) / (rewards_normalized.std() + 1e-5)

        # Optimization steps
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            action_probs, state_values = self.policy(states)
            dist = Categorical(action_probs)

            log_probs = dist.log_prob(actions)
            dist_entropy = dist.entropy()
            state_values = torch.squeeze(state_values)

            # Ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(log_probs - old_log_probs)

            # Surrogate Loss
            advantages = rewards_normalized - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # Loss = -min(surr1, surr2) + 0.5*MSE(val) - 0.01*Entropy
            loss = -torch.min(surr1, surr2) + 0.5*F.mse_loss(state_values, rewards_normalized) - 0.01*dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.data = [] # Clear memory

# ==============================================================================
# 4. MAIN TRAINING LOOP
#