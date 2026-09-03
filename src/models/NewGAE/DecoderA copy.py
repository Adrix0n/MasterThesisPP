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

		self.node_generator = nn.Linear(current_in_features, int(max_nodes * self.node_embed_dim))
		self.edge_predictor = nn.Linear(self.node_embed_dim, 1)

	def forward(self, z: torch.Tensor) -> torch.Tensor:
		batch_size = z.size(0)
		x = z

		for dense in self.denses:
			x = dense(x)

		x = self.node_generator(x)

		node_embeddings = x.view(batch_size, self.max_nodes, self.node_embed_dim)

		nodes_i = node_embeddings.unsqueeze(2)  # Kształt: [Batch, Nodes, 1, Features]
		nodes_j = node_embeddings.unsqueeze(1)  # Kształt: [Batch, 1, Nodes, Features]


		pair_features = nodes_i * nodes_j  # Kształt: [Batch, Nodes, Nodes, Features]

		adj_logits = self.edge_predictor(pair_features)  # Kształt: [Batch, Nodes, Nodes, 1]
		adj_logits = adj_logits.squeeze(-1) 

		return adj_logits