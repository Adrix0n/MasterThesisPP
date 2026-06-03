import torch
import torch.nn as nn
from torch_geometric.nn import DenseGCNConv

class ConvLayer(nn.Module):
	"""
	Blok konwolucyjny
	"""

	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			activation: str = 'relu',
			use_norm: bool = True,
			dropout_rate: float = 0.0
	):
		super().__init__()

		# TODO: Dodać inne GConv
		self.conv = DenseGCNConv(in_channels, out_channels)

		if use_norm:
			self.norm = nn.LayerNorm(out_channels)
		else:
			self.norm = nn.Identity()

		activation_lower = activation.lower()
		if activation_lower == 'relu':
			self.act = nn.ReLU()
		elif activation_lower == 'gelu':
			self.act = nn.GELU()
		else:
			self.act = nn.Identity()

		if dropout_rate > 0.0:
			self.dropout = nn.Dropout(dropout_rate)
		else:
			self.dropout = nn.Identity()

	def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
		"""
		- x: Macierz cech o kształcie [batch_size, node_size, features]
		- adj: Macierz sąsiedztwa o kształcie [batch_size, node_size, node_size]
		"""
		x = self.conv(x, adj)
		x = self.norm(x)
		x = self.act(x)
		x = self.dropout(x)

		return x