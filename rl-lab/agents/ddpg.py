
# Add the relative or absolute path to the folder you need to import from
sys.path.append(os.path.abspath("../"))
import sys
import os

import random
import collections
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from envs.ceviche.env import CevicheEnv



# ================================
# Hyperparameters
# ================================
ENV_NAME = "Ceviche-Design-Space"   # Environment name (continuous action space)
GAMMA = 0.99               # Discount factor for future rewards
ACTOR_LR = 1e-4            # Learning rate for the actor network
CRITIC_LR = 1e-3           # Learning rate for the critic network
BUFFER_SIZE = 100000       # Maximum size of the replay buffer
BATCH_SIZE = 64            # Mini-batch size for sampling from the replay buffer
TAU = 1e-3                 # Soft update factor for target networks
NOISE_SCALE = 0.1          # Scale of Ornstein-Uhlenbeck exploration noise
MAX_EPISODES = 500         # Total number of training episodes
MAX_STEPS = 200            # Maximum steps per episode

# ================================
# Replay Buffer Definition
# ================================
Transition = collections.namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity):
        # Use a deque to automatically discard old experiences when full
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, *args):
        """
        Save a transition (state, action, reward, next_state, done).
        Each call appends to the deque.
        """
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from the buffer.
        Returns a Transition namedtuple of batches.
        """
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        """Return current size of the buffer."""
        return len(self.buffer)

# ================================
# Actor Network Definition
# ================================
class Actor(nn.Module):
    """
    Actor network maps state to a deterministic action in [0, 1].
    Uses sigmoid activation on the final layer to ensure outputs lie between 0 and 1.
    state_dim: size of the input vector (int)
    action_dime: size of the output vector/predictions (int)
    """
    # State dimensions: 3600. Each number corresponds to a pixel in the 
    #   original design space. 
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Define a 3-layer feedforward network
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),   # Input: state_dim -> 400
            nn.ReLU(),                   # Non-linear activation
            nn.Linear(400, 300),         # Hidden: 400 -> 300
            nn.ReLU(),                   # Non-linear activation
            nn.Linear(300, action_dim),  # Output: 300 -> action_dim
            nn.Sigmoid()                 # Sigmoid to bound actions in [0, 1]
        )

    def forward(self, x):
        """
        Forward pass: compute action in [0, 1].
        Input: x (state) with shape [batch_size, state_dim]
        Output: action with shape [batch_size, action_dim], each in [0, 1]
        """
        return self.net(x)

# ================================
# Critic Network Definition
# ================================
class Critic(nn.Module):
    """
    Critic network maps (state, action) pairs to a Q-value.
    It concatenates state and action, then passes through hidden layers.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Define a 3-layer feedforward network that takes (state + action)
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),  # Input: state_dim + action_dim -> 400
            nn.ReLU(),                               # Non-linear activation
            nn.Linear(400, 300),                     # Hidden layer: 400 -> 300
            nn.ReLU(),                               # Non-linear activation
            nn.Linear(300, 1)                        # Output: Single Q-value
        )

    def forward(self, state, action):
        """
        Forward pass: concatenate state and action and compute Q-value.
        Inputs:
            - state: tensor of shape [batch_size, state_dim]
            - action: tensor of shape [batch_size, action_dim]
        Output:
            - Q-value: tensor of shape [batch_size, 1]
        """
        x = torch.cat([state, action], dim=1)  # Concatenate along the feature dimension
        return self.net(x)

# ================================
# Ornstein-Uhlenbeck Noise for Exploration
# ================================
class OUNoise:
    """
    Ornstein-Uhlenbeck process for temporally correlated exploration noise.
    Adds noise to actions for exploration in continuous action spaces.
    """

    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu          # Long-running mean
        self.theta = theta    # Speed of mean reversion
        self.sigma = sigma    # Volatility (noise scale)
        # Initialize internal state to the mean
        self.state = np.ones(action_dim) * self.mu

    def reset(self):
        """Reset the internal state to the mean."""
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        """
        Generate noise using Ornstein-Uhlenbeck dynamics:
        dx = theta * (mu - x) + sigma * N(0, 1)
        x <- x + dx
        """
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

# ================================
# Soft Update for Target Networks
# ================================
def soft_update(target, source, tau):
    """
    Soft-update: target <- tau*source + (1-tau)*target
    Ensures target network changes slowly for stability.
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

# ================================
# Main DDPG Training Loop
# ================================
def train():
    env = CevicheEnv()  # Create environment
    state_dim = (env.Nx // 2) * (env.Ny // 2)  # Dimensions are 60x60 from the design space
    action_dim = state_dim     # Design choice: update the entire design space, parameterize percentage later
    max_action = float(env.action_space.high[0])  # Action range: [0, 1]

    # Instantiate online networks
    actor = Actor(state_dim, action_dim, max_action)
    critic = Critic(state_dim, action_dim)
    # Instantiate target networks, copying weights from online networks
    target_actor = Actor(state_dim, action_dim, max_action)
    target_critic = Critic(state_dim, action_dim)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    # Optimizers for actor and critic
    actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    # Initialize replay buffer and noise process
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    noise = OUNoise(action_dim)

    # Pre-fill replay buffer with random actions to ensure enough samples
    state = env.reset()
    for _ in range(BUFFER_SIZE // 10):
        # Random action for initial exploration

        # action_space is currently not a method of env, could probably
        # just use untrained model since this is initialized randomly
        # generate a random state
        # call actor on state
        next_state, reward = env.step(action)
        done = False # manually updating for now
        replay_buffer.push(state, action, reward, next_state, done)
        # If episode ends, reset environment
        state = next_state if not done else env.reset()

    # Training loop over episodes
    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()   # Reset at the start of each episode
        noise.reset()         # Reset exploration noise
        episode_reward = 0

        for step in range(MAX_STEPS):
            # Convert state to tensor for the actor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Shape: [1, state_dim]
            # Get deterministic action from actor, then add exploration noise
            action = actor(state_tensor).detach().cpu().numpy()[0]
            action = action + noise.sample() * NOISE_SCALE
            # Clip action to valid range
            action = np.clip(action, 0, max_action)

            # Step environment and collect feedback
            next_state, reward = env.step(action)
            # Store transition in replay buffer
            done = False
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward # Design choice

            # Only update if buffer has enough samples
            if len(replay_buffer) >= BATCH_SIZE:
                # Sample a random mini-batch of transitions
                batch = replay_buffer.sample(BATCH_SIZE)
                states = torch.FloatTensor(batch.state)           # [batch_size, state_dim]
                actions = torch.FloatTensor(batch.action).unsqueeze(1)  # [batch_size, action_dim]
                rewards = torch.FloatTensor(batch.reward).unsqueeze(1)  # [batch_size, 1]
                next_states = torch.FloatTensor(batch.next_state) # [batch_size, state_dim]
                dones = torch.FloatTensor(batch.done).unsqueeze(1)      # [batch_size, 1]

                # -------------------------
                # Critic Update (TD target)
                # -------------------------
                with torch.no_grad():
                    # Compute next action from target actor
                    next_actions = target_actor(next_states)
                    # Compute target Q-value from target critic
                    target_q = target_critic(next_states, next_actions)
                    # TD target: r + gamma * Q_target(s', mu_target(s')) * (1 - done)
                    y = rewards + GAMMA * (1 - dones) * target_q

                # Current Q estimate
                current_q = critic(states, actions)
                # Critic loss (mean squared error between current and target Q)
                critic_loss = nn.MSELoss()(current_q, y)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # -------------------------
                # Actor Update (Policy)
                # -------------------------
                actor_optimizer.zero_grad()
                # Actor loss: maximize Q value (equiv. minimize negative Q)
                actor_loss = -critic(states, actor(states)).mean()
                actor_loss.backward()
                actor_optimizer.step()

                # -------------------------
                # Soft update target networks
                # -------------------------
                soft_update(target_actor, actor, TAU)
                soft_update(target_critic, critic, TAU)

            # End episode if done
            if done:
                break

        # Print episode result
        print(f"Episode {episode}: Reward = {episode_reward:.2f}")

    # Close environment and save models
    env.close()
    torch.save(actor.state_dict(), "ddpg_actor.pth")
    torch.save(critic.state_dict(), "ddpg_critic.pth")

# ================================
# Run training
# ================================
if __name__ == "__main__":
    train()
