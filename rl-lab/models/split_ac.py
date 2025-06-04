import gym
from split_mlp import SplitMLP
import torch
import torch.nn as nn

""" 
Class to define actor critic model.  
"""
class SplitActorCritic(nn.Module):
	def __init__(
		self,
		split_actor_net: SplitMLP,
		critic_net: nn.Module,
		cont_action_space,
		disc_action_space
	):
        
		super().__init__()
		self.split_actor = split_actor_net
		self.critic = critic_net
		
		self.split = split_actor_net.split
		self.output_dim = split_actor_net.output_dim
		
		self.cont_action_space = cont_action_space
		self.disc_action_space = disc_action_space
		
        # Load continuous out
		self.mu = nn.Linear(self.split, cont_action_space.shape[0])
		self.log_std = nn.Parameter(torch.zeros(cont_action_space.shape[0]))
		
        # Load disc out
		self.logits = nn.Linear(self.output_dim - self.split, disc_action_space.n)
				
		self.v_net = nn.Linear(critic_net.output_dim, 1)
	

	def forward(self, x):
		a_out = self.split_actor(x)
		c_out = self.critic(x)
		
		mean = self.mu(a_out[:,:self.split])
		std = torch.exp(self.log_std)
		cont_dist = torch.distributions.Normal(mean, std)
		
		logits = self.logits(a_out[:, self.split:])
		disc_dist = torch.distributions.Categorical(logits=logits)
		
		value = self.v_net(c_out).squeeze(-1)
		
		return cont_dist, disc_dist, value
	


