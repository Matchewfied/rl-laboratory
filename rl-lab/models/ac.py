import gym
import torch
import torch.nn as nn

""" 
Class to define actor critic model.  
"""
class ActorCritic(nn.Module):
	def __init__(
		self,
		actor_net: nn.Module,
		critic_net: nn.Module,
		action_space,
		is_continuous=False
	):

		super().__init__()
		self.actor = actor_net
		self.critic = critic_net
		self.action_space = action_space

		if is_continuous:
			self.mu = nn.Linear(actor_net.output_dim, action_space.shape[0])
			self.log_std = nn.Parameter(torch.zeros(action_space.shape[0]))
		else:
			self.logits = nn.Linear(actor_net.output_dim, action_space.n)
				
		self.v_net = nn.Linear(critic_net.output_dim, 1)


	def forward(self, x):
		a_out = self.actor(x)
		c_out = self.critic(x)

		if self.continuous:
			mean = self.mu(a_out)
			std = torch.exp(self.log_std)
			dist = torch.distributions.Independent(torch.distributions.Normal(mean, std), 1)
		else:
			logits = self.logits(a_out)
			dist = torch.distributions.Categorical(logits=logits)
		
		value = self.v_net(c_out).squeeze(-1)

		return dist, value


