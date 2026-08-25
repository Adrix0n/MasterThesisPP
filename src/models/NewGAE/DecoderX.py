import torch
import torch.nn as nn
from typing import List, Dict, Any

from src.models.NewGAE.DenseLayer import DenseLayer


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
		self.final_projection = nn.Linear(current_in_features, max_nodes * num_features)

	def forward(self, z: torch.Tensor) -> torch.Tensor:
		batch_size = z.size(0)
		x = z

		for dense in self.denses:
			x = dense(x)

		x = self.final_projection(x)

		x_prime = x.view(batch_size, self.max_nodes, self.num_features)

		return x_prime