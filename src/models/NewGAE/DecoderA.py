import torch
import torch.nn as nn
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
		self.edge_weights = nn.Parameter(torch.ones(self.node_embed_dim))
		self.edge_bias = nn.Parameter(torch.zeros(1))

	def forward(self, z: torch.Tensor) -> torch.Tensor:
		batch_size = z.size(0)
		x = z

		for dense in self.denses:
			x = dense(x)

		x = self.node_generator(x)

		node_embeddings = x.view(batch_size, self.max_nodes, self.node_embed_dim)
		scaled_embeddings = node_embeddings * self.edge_weights

		adj_logits = torch.bmm(node_embeddings, scaled_embeddings.transpose(1, 2).contiguous())
		adj_logits = adj_logits + self.edge_bias
		adj_logits = torch.clamp(adj_logits, min=-20.0, max=20.0)

		return adj_logits