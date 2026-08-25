import torch
import torch.nn as nn
from typing import List
from src.models.KlejdaGAE.KlejdaConvLayer import KlejdaConvLayer
from src.models.KlejdaGAE.KlejdaDenseLayer import KlejdaDenseLayer

class KlejdaEncoder(nn.Module):
	"""
	Enkoder
	"""

	def __init__(
			self,
			in_channels: int,
			conv_channels: List[int],
			dense_features: List[int],
			max_nodes: int
	):
		super().__init__()
		self.max_nodes = max_nodes

		self.convs = nn.ModuleList()
		current_in_channels = in_channels

		for out_channels in conv_channels:
			self.convs.append(
				KlejdaConvLayer(current_in_channels, out_channels)
			)
			current_in_channels = out_channels

		self.flatten_size = max_nodes * current_in_channels

		self.denses = nn.ModuleList()
		current_in_features = self.flatten_size

		for out_features in dense_features:
			self.denses.append(
				KlejdaDenseLayer(current_in_features, out_features)
			)
			current_in_features = out_features

		self.output_dim = current_in_features

	def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
		batch_size = x.size(0)

		for conv in self.convs:
			x = conv(x, adj)

		x = x.view(batch_size, -1)

		for dense in self.denses:
			x = dense(x)
		return x