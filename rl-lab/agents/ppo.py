from buffers.rollout import RolloutBuffer
import torch


class PPOAgent:
    def __init__(self, ac_model, optimizer, config, device='cpu'):
        self.model = ac_model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.buffer = RolloutBuffer(config.buffer_size, config.obs_shape, config.action_shape, device)
        
    def select_action(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dype=torch.float32)
        obs = obs.to(self.device)

        dist, value = self.model(obs)
        
        # Should be of shape (batch_dim, action_dim)
        action = dist.sample()

        # Should be of shape (batch_dim,)
        logprob = dist.log_prob(action)

        return action.cpu().numpy(), logprob, value.item() 


    def store_transition(self, obs, action, reward, done, logprob, value):
        self.buffer.add(obs, action, reward, done, logprob, value)

    def train(self):
        # batch -> get batch
        # ppo_update ...
        # clear buff
        

    def save(self, path):
        torch.save(self.model.state_dict(), path)