import torch.nn as nn
import torch


class DenseLayer(nn.Module):
	"""
	Blok warstwy gęstej
	"""

	def __init__(
			self,
			in_features: int,
			out_features: int,
			activation: str = 'silu',
			use_norm: bool = True,
			dropout_rate: float = 0.0):
		super().__init__()

		self.linear = nn.Linear(in_features=in_features, out_features=out_features)

		if use_norm:
			self.norm = nn.BatchNorm1d(out_features)
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
			self.act = nn.SiLU()
		else:
			raise ValueError(
				f"Nieobsługiwana funkcja aktywacji: {activation}. Wybierz spośród: relu, gelu, elu, tanh, none."
			)

		if dropout_rate > 0.0:
			if not (0.0 <= dropout_rate < 1.0):
				raise ValueError("Współczynnik dropout_rate musi zawierać się w przedziale [0, 1).")
			self.dropout = nn.Dropout(dropout_rate)
		else:
			self.dropout = nn.Identity()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.linear(x)
		x = self.norm(x)
		x = self.act(x)
		x = self.dropout(x)

		return x