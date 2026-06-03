import torch
import torch.nn as nn
from typing import List
from src.models.KlejdaGAE.ConvLayer import ConvLayer
from src.models.KlejdaGAE.DenseLayer import DenseLayer

class Encoder(nn.Module):
	"""
	Enkoder
	"""

	def __init__(
			self,
			in_channels: int,
			conv_channels: List[int],
			dense_features: List[int],
			max_nodes: int = 15
	):
		super().__init__()
		self.max_nodes = max_nodes

		self.convs = nn.ModuleList()
		current_in_channels = in_channels

		for out_channels in conv_channels:
			self.convs.append(
				ConvLayer(current_in_channels, out_channels)
			)
			current_in_channels = out_channels

		self.flatten_size = max_nodes * current_in_channels

		self.denses = nn.ModuleList()
		current_in_features = self.flatten_size

		for out_features in dense_features:
			self.denses.append(
				DenseLayer(current_in_features, out_features)
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