import torch
import torch.nn as nn
from typing import List, Dict, Any

from src.models.NewGAE.DenseLayer import DenseLayer
from src.models.NewGAE.ConvLayer import ConvLayer


class DecoderX(nn.Module):
	def __init__(
			self,
			latent_dim: int,
			hidden_dims: List[int],
			max_nodes: int,
			num_features: int,
			config: Dict[str, Any]
	):
		super().__init__()
		self.max_nodes = max_nodes
		self.num_features = num_features

		self.denses = nn.ModuleList()
		current_in_features = latent_dim
		for out_features in hidden_dims:
			self.denses.append(
				DenseLayer(
					in_features=current_in_features,
					out_features=out_features,
					activation=config['dense_activation'],
					use_norm=config['dense_use_norm'],
					dropout_rate=config['dense_dropout_rate']
				)
			)
			current_in_features = out_features

		conv_channels = config['decoder_x_conv_channels']
		initial_node_dim = conv_channels[0]

		self.global_to_nodes = nn.Linear(
			current_in_features,
			max_nodes * initial_node_dim
		)

		self.convs = nn.ModuleList()
		current_in_channels = initial_node_dim
		for out_channels in conv_channels:
			self.convs.append(
				ConvLayer(
					in_channels=current_in_channels,
					out_channels=out_channels,
					activation=config['conv_activation'],
					use_norm=config['conv_use_norm'],
					dropout_rate=config['conv_dropout_rate'],
					conv_type=config['conv_type'],
				)
			)
			current_in_channels = out_channels

		self.final_projection = nn.Linear(current_in_channels, num_features)

	def forward(self, z: torch.Tensor, a_prime: torch.Tensor) -> torch.Tensor:
		batch_size = z.size(0)
		x = z
		for dense in self.denses:
			x = dense(x)

		x = self.global_to_nodes(x)
		x = x.view(batch_size, self.max_nodes, -1)  # Kształt: [batch_size, max_nodes, initial_node_dim]

		for conv in self.convs:
			x = conv(x, a_prime)

		x_prime = self.final_projection(x)

		return x_prime