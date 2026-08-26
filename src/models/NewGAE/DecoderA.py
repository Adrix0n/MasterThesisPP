import torch
import torch.nn as nn
import math
from typing import List, Any, Dict
from src.models.NewGAE.DenseLayer import DenseLayer


class DecoderA(nn.Module):
	def __init__(
			self,
			latent_dim: int,
			hidden_dims: List[int],
			max_nodes: int,
			config: Dict[str, Any]
	):
		super().__init__()
		self.max_nodes = max_nodes
		self.node_embed_dim = config['decoder_mlp_out_features']

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

		self.node_generator = nn.Linear(current_in_features, max_nodes * self.node_embed_dim)
		self.node_pos_embed = nn.Parameter(torch.randn(1, max_nodes, self.node_embed_dim) * 0.05)

	def forward(self, z: torch.Tensor) -> torch.Tensor:
		batch_size = z.size(0)
		x = z

		for dense in self.denses:
			x = dense(x)

		x = self.node_generator(x)

		node_embeddings = x.view(batch_size, self.max_nodes, self.node_embed_dim)
		node_embeddings = node_embeddings + self.node_pos_embed

		# Generowanie logitów macierzy sąsiedztwa (symetrycznej)
		adj_logits = torch.bmm(node_embeddings, node_embeddings.transpose(1, 2))
		adj_logits = adj_logits / math.sqrt(self.node_embed_dim)

		# Zwracane są logity, sigmoida nakładana będzie później
		return adj_logits