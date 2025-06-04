import torch
import torch.nn as nn

class SplitMLP(nn.Module):

	def __init__(
		self,
		base: nn.Module,
		cont_head: nn.Module,
		disc_head: nn.Module
	):

		super().__init__()
		self.base = base
		self.cont_head = cont_head
		self.disc_head = disc_head
		self.split = cont_head.output_dim
		self.output_dim = cont_head.output_dim + disc_head.output_dim

	
	def forward(self, x):
		y = self.base(x)
		c_out = self.cont_head(y)
		d_out = self.disc_head(y)

		return torch.cat([c_out, d_out], dim=-1)
		