import torch
import torch.nn as nn
from typing import List, Dict, Any

from src.models.NewGAE.DenseLayer import DenseLayer
from src.models.NewGAE.ConvLayer import ConvLayer


class Encoder(nn.Module):
	def __init__(
			self,
			in_channels: int,
			conv_channels: List[int],
			dense_features: List[int],
			max_nodes: int,
			use_mlp: bool,
			config: Dict[str, Any],
	):
		super().__init__()
		self.max_nodes = max_nodes
		self.use_mlp = use_mlp

		if use_mlp:
			mlp_out_features = config['encoder_mlp_out_features']
			self.mlp = nn.Linear(in_channels, mlp_out_features)
			current_in_channels = mlp_out_features
		else:
			self.mlp = nn.Identity()
			current_in_channels = in_channels

		self.convs = nn.ModuleList()

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

		self.denses = nn.ModuleList()
		current_in_features = current_in_channels
		for out_features in dense_features:
			self.denses.append(
				DenseLayer(
					in_features = current_in_features,
					out_features = out_features,
					activation = config['dense_activation'],
					use_norm = config['dense_use_norm'],
					dropout_rate = config['dense_dropout_rate']
				)
			)
			current_in_features = out_features

		self.output_dim = current_in_features

	def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
		x = self.mlp(x)

		for conv in self.convs:
			x = conv(x, adj)

		x = x.mean(dim=1)

		for dense in self.denses:
			x = dense(x)

		return x