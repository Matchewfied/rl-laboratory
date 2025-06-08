import random
import collections
import numpy as np
import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from ..envs.ceviche.env import CevicheEnv



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
NOISE_SCALE = 0.5          # Scale of Ornstein-Uhlenbeck exploration noise
MAX_EPISODES = 500         # Total number of training episodes
MAX_STEPS = 50            # Maximum steps per episode

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
    def __init__(self, state_dim, action_dim, net):
        super().__init__()
        # # Define a 3-layer feedforward network
        # self.net = nn.Sequential(
        #     nn.Linear(state_dim, 400),   # Input: state_dim -> 400
        #     nn.ReLU(),                   # Non-linear activation
        #     nn.Linear(400, 300),         # Hidden: 400 -> 300
        #     nn.ReLU(),                   # Non-linear activation
        #     nn.Linear(300, action_dim),  # Output: 300 -> action_dim
        #     nn.Sigmoid()                 # Sigmoid to bound actions in [0, 1]
        # )
        self.net = net

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

def stack_tensors(data, from_numpy=False):
    if from_numpy:
        return torch.stack([torch.from_numpy(arr) for arr in data])
    
    if data[0].ndim > 1:
        return torch.stack([arr[0] for arr in data])
    return torch.stack([arr for arr in data])
 
def train(num_episodes, num_steps, save_dir, save_rho, net, experiment_name): # Can turn this into a config eventually.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CevicheEnv(save_dir=save_dir, save_rho=save_rho, experiment_name=experiment_name)
    state_dim = (env.Nx // 2) * (env.Ny // 2)
    action_dim = state_dim
    max_action = 1.0

    # Models to GPU
    actor = Actor(state_dim, action_dim, net).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    target_actor = Actor(state_dim, action_dim, net).to(device)
    target_critic = Critic(state_dim, action_dim).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    noise = OUNoise(action_dim)

    # Prefill buffer
    total_steps = 0
    state = env.reset()
    for _ in range(BUFFER_SIZE // 1000):
        state_t = torch.from_numpy(state).float().to(device)
        action = actor(state_t.unsqueeze(0)).detach().cpu()
        action = action.view(1, -1)
        next_state, reward = env.step([action[0].numpy(), np.arange(3600)])
        done = False
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        noise.reset()
        episode_reward = 0

        for step in range(num_steps):
            total_steps += 1
            state_tensor = torch.FloatTensor(state).to(device)
            action = actor(state_tensor.unsqueeze(0)).detach().cpu()
            action = action + noise.sample() * NOISE_SCALE
            action = action.view(1, -1)
            action = torch.clamp(action, 0, max_action)

            next_state, reward = env.step([action[0].numpy(), np.arange(3600)])
            done = False
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(replay_buffer) >= BATCH_SIZE:
                batch = replay_buffer.sample(BATCH_SIZE)
                states = stack_tensors(batch.state, from_numpy=True).float().to(device)
                actions = stack_tensors(batch.action, from_numpy=False).float().to(device)
                rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(batch.next_state).to(device)
                dones = torch.FloatTensor(batch.done).unsqueeze(1).to(device)

                with torch.no_grad():
                    next_actions = target_actor(next_states)
                    target_q = target_critic(next_states, next_actions)
                    y = rewards + GAMMA * (1 - dones) * target_q

                critic_optimizer.zero_grad()
                current_q = critic(states, actions)
                critic_loss = nn.MSELoss()(current_q, y)
                critic_loss.backward()
                critic_optimizer.step()

                actor_optimizer.zero_grad()
                predicted_actions = actor(states)
                actor_loss = -critic(states, predicted_actions).mean()
                actor_loss.backward()
                actor_optimizer.step()

                soft_update(target_actor, actor, TAU)
                soft_update(target_critic, critic, TAU)

            if done:
                break

        print(f"Episode {episode}: Reward = {episode_reward:.2f}")
        if total_steps % 2000 == 1999:
            actor_save_path = os.path.join(save_dir, "ddpg_actor.pth")
            critic_save_path = os.path.join(save_dir, "ddpg_critic.pth")
            torch.save(actor.state_dict(), actor_save_path)
            torch.save(critic.state_dict(), critic_save_path)
   



    env.close()

# ================================
# Run training
# ================================
if __name__ == "__main__":
    train()
