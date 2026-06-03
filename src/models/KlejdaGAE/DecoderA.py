import torch
import torch.nn as nn
from typing import List
from src.models.KlejdaGAE.DenseLayer import DenseLayer

class DecoderA(nn.Module):
	"""
	Dekoder odtwarzający macierz sąsiedztwa A'.
	"""

	def __init__(
			self,
			latent_dim: int,
			hidden_dims: List[int],
			max_nodes: int = 15
	):
		super().__init__()
		self.max_nodes = max_nodes

		self.denses = nn.ModuleList()
		current_in_features = latent_dim

		for out_features in hidden_dims:
			self.denses.append(
				DenseLayer(
					in_features=current_in_features,
					out_features=out_features,
					activation='relu'
				)
			)
			current_in_features = out_features

		self.final_projection = nn.Linear(current_in_features, max_nodes * max_nodes)

		# Aktywacja Sigmoid wymuszająca wartości z przedziału (0,1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, z: torch.Tensor) -> torch.Tensor:
		batch_size = z.size(0)
		x = z

		for dense in self.denses:
			x = dense(x)
		x = self.final_projection(x)
		x = self.sigmoid(x)

		a_prime = x.view(batch_size, self.max_nodes, self.max_nodes)
		return a_prime