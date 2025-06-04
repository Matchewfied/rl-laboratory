import numpy as np
import torch



class RolloutBuffer:
    """ Rollout buffer used by the PPOAgent.
    
    Args:
        (to fill out)

    """
    def __init__(self, size, obs_shape, action_shape,
                 device='cpu', gamma=0.99, gae_lambda=0.95, dtype=np.float32):
        self.size = size
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.dtype = dtype

        self.obs = np.zeros((size, *obs_shape), dtype=dtype)
        self.actions = np.zeros((size, *action_shape), dtype=dtype)
        self.rewards = np.zeros(size, dtype=dtype)
        self.dones = np.zeros(size, dtype=dtype)
        self.logprobs = np.zeros(size, dtype=dtype)
        self.values = np.zeros(size, dtype=dtype)

        self.advantages = np.zeros(size, dype=dtype)
        self.returns = np.zeros(size, dtype=dtype)

        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, done, logprob, value):
        idx = self.idx

        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.logprobs[idx] = logprob
        self.values[idx] = value

        self.idx += 1
        if self.idx == self.size:
            self.full = True
            self.idx = 0
    
    def compute_advantages(self, last_value, done):
        adv = 0
        for step in reversed(range(self.size)):
            if step == self.size - 1:
                next_non_terminal = 1.0 - done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            

            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            adv = delta + self.gamma * self.gae_lambda * next_non_terminal * adv

        self.returns = self.advantages + self.values

    def get_batch(self, batch_size, shuffle=True):
        N = len(self.obs)
        indices = np.arange(N)

        if shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, N, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield {
                'observations': self.obs[batch_indices],
                'actions': self.actions[batch_indices],
                'log_probs': self.logprobs[batch_indices],
                'returns': self.returns[batch_indices],
                'advantages': self.advantages[batch_indices],
            }

    def reset(self):
        self.idx = 0
        self.full = False