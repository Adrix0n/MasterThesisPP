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

		# 1. Przygotowanie wektorów do mnożenia par
		# Rozszerzamy wymiary, aby stworzyć macierz par
		nodes_i = node_embeddings.unsqueeze(2)  # Kształt: [Batch, Nodes, 1, Features]
		nodes_j = node_embeddings.unsqueeze(1)  # Kształt: [Batch, 1, Nodes, Features]

		# 2. Iloczyn Hadamarda (element po elemencie)
		# Broadcasting PyTorcha automatycznie stworzy kombinację każdego węzła z każdym.
		# Z natury A * B == B * A, więc struktura jest w 100% symetryczna.
		pair_features = nodes_i * nodes_j  # Kształt: [Batch, Nodes, Nodes, Features]

		# 3. Ewaluacja par (generowanie logitów macierzy)
		# Warstwa Linear działa na ostatnim wymiarze (Features), zwracając jedno prawdopodobieństwo
		adj_logits = self.edge_predictor(pair_features)  # Kształt: [Batch, Nodes, Nodes, 1]
		adj_logits = adj_logits.squeeze(-1) 

		return adj_logits