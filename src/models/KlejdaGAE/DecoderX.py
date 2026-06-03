import torch
import torch.nn as nn
from typing import List
from src.models.KlejdaGAE.DenseLayer import DenseLayer
from src.models.KlejdaGAE.ConvLayer import ConvLayer

class DecoderX(nn.Module):
	"""
	Dekoder odtwarzający macierz X'
	"""

	def __init__(
			self,
			latent_dim: int,
			conv_channels: List[int],
			dense_features: List[int],
			out_features: int = 3,
			max_nodes: int = 15
	):
		super().__init__()
		self.max_nodes = max_nodes
		self.out_features = out_features

		self.z_to_nodes = nn.Linear(latent_dim, latent_dim * max_nodes)

		self.convs = nn.ModuleList()
		current_in_channels = latent_dim

		for out_channels in conv_channels:
			self.convs.append(
				ConvLayer(
					in_channels=current_in_channels,
					out_channels=out_channels,
					activation='relu'
				)
			)
			current_in_channels = out_channels

		self.flatten_size = max_nodes * current_in_channels

		self.denses = nn.ModuleList()
		current_in_features = self.flatten_size

		for out_feat in dense_features:
			self.denses.append(
				DenseLayer(
					in_features=current_in_features,
					out_features=out_feat,
					activation='relu'
				)
			)
			current_in_features = out_feat

		self.final_projection = nn.Linear(current_in_features, max_nodes * out_features)
		self.final_relu = nn.ReLU()

	def forward(self, z: torch.Tensor, a_prime: torch.Tensor) -> torch.Tensor:
		batch_size = z.size(0)
		x = self.z_to_nodes(z)
		x = x.view(batch_size, self.max_nodes, -1)

		for conv in self.convs:
			x = conv(x, a_prime)

		x = x.view(batch_size, -1)

		for dense in self.denses:
			x = dense(x)

		x = self.final_projection(x)
		x = self.final_relu(x)

		x_prime = x.view(batch_size, self.max_nodes, self.out_features)
		x_prime = x_prime - 1.0
		return x_prime