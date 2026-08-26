import torch
import torch.nn as nn
from torch_geometric.nn import DenseGCNConv, DenseGINConv
from src.models.NewGAE.DenseGATConv import DenseGATConv
class ConvLayer(nn.Module):
	"""
	Blok konwolucyjny
	"""

	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			activation: str,
			use_norm: bool,
			dropout_rate: float,
			conv_type: str):
		super().__init__()

		conv_type_lower = conv_type.lower()
		if conv_type_lower == 'dense_gcn_conv':
			self.conv = DenseGCNConv(in_channels, out_channels)
		elif conv_type_lower == 'dense_gin_conv':
			mlp = nn.Sequential(
				nn.Linear(in_channels, out_channels),
				nn.ReLU(),
				nn.Linear(out_channels, out_channels)
			)
			self.conv = DenseGINConv(mlp)
		elif conv_type_lower == 'dense_gat_conv':
			self.conv = DenseGATConv(in_channels, out_channels, dropout_rate)
		else:
			raise ValueError(
				f"Nieobsługiwana warstwa konwolucyjna: {conv_type}. Dostępne: dense_gcn_conv, dense_gin_conv, dense_gat_conv.")

		if use_norm:
			self.norm = nn.BatchNorm1d(out_channels)
		else:
			self.norm = nn.Identity()

		activation_lower = activation.lower()
		if activation_lower == 'relu':
			self.act = nn.ReLU()
		elif activation_lower == 'gelu':
			self.act = nn.GELU()
		elif activation_lower == 'elu':
			self.act = nn.ELU()
		elif activation_lower == 'tanh':
			self.act = nn.Tanh()
		elif activation_lower in ['none', 'linear']:
			self.act = nn.Identity()
		elif activation_lower == 'leaky_relu':
			self.act = nn.LeakyReLU()
		elif activation_lower == 'silu':
			# W fizyce i geometrii to Twój faworyt!
			self.act = nn.SiLU()
		else:
			raise ValueError(
				f"Nieobsługiwana funkcja aktywacji: {activation}. Dostępne: relu, gelu, elu, tanh, none, leaky_relu, silu."
			)

		if dropout_rate > 0.0:
			if not (0.0 <= dropout_rate < 1.0):
				raise ValueError("dropout_rate musi zawierać się w przedziale [0, 1).")
			self.dropout = nn.Dropout(dropout_rate)
		else:
			self.dropout = nn.Identity()

	def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
		x = self.conv(x, adj)
		x = x.transpose(1, 2)  # (Batch, Nodes, Features) -> (B, Features, Nodes)
		x = self.norm(x)
		x = x.transpose(1, 2)  # Back to (Batch, Nodes, Features)
		x = self.act(x)
		return x