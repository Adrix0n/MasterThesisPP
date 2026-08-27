import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from src.models.NewGAE.Encoder import Encoder
from src.models.NewGAE.DecoderA import DecoderA
from src.models.NewGAE.DecoderX import DecoderX
from src.models.BaseGraphAutoEncoder import BaseGraphAutoEncoder


class GraphAutoencoder(BaseGraphAutoEncoder):

	def __init__(self, config: Dict[str, Any], frams_module):
		super().__init__(config, frams_module)

		# Inicjalizacja Enkodera
		self.encoder_backbone = Encoder(
			in_channels=self.hparams.in_channels,
			conv_channels=self.hparams.encoder_conv_channels,
			dense_features=self.hparams.encoder_dense_features,
			max_nodes=self.hparams.max_nodes,
			use_mlp=self.hparams.get('encoder_use_mlp', True),
			config=config
		)

		# Warstwa rzutująca do przestrzeni ukrytej Z
		self.fc_z = nn.Linear(self.encoder_backbone.output_dim, self.hparams.latent_dim)

		# Inicjalizacja Dekodera A
		self.decoder_a = DecoderA(
			latent_dim=self.hparams.latent_dim,
			hidden_dims=self.hparams.decoder_a_hidden_dims,
			max_nodes=self.hparams.max_nodes,
			config=config
		)

		# Inicjalizacja Dekodera X
		self.decoder_x = DecoderX(
			latent_dim=self.hparams.latent_dim,
			hidden_dims=self.hparams.decoder_x_dense_features,
			max_nodes=self.hparams.max_nodes,
			num_features=self.hparams.in_channels,
			config=config
		)

		self.apply(self._init_weights)

	def forward(self, x: torch.Tensor, adj: torch.Tensor):
		hidden_features = self.encoder_backbone(x, adj)
		z = self.fc_z(hidden_features)

		a_logits = self.decoder_a(z)
		a_probs = torch.sigmoid(a_logits)

		x_prime = self.decoder_x(z, a_probs)

		return x_prime, a_logits, z

	def encode(self, x: torch.Tensor, adj: torch.Tensor):
		hidden_features = self.encoder_backbone(x, adj)
		z = self.fc_z(hidden_features)
		return z

	def decode(self, z: torch.Tensor):
		a_logits = self.decoder_a(z)
		a_probs = torch.sigmoid(a_logits)
		x_prime = self.decoder_x(z, a_probs)

		return x_prime, a_probs

	def compute_reconstruction_loss(self, batch):
		x, adj, properties = batch
		parts_num = properties['parts_num']

		x_prime, a_logits, z = self.forward(x, adj)

		# Maski
		node_mask = torch.arange(self.hparams.max_nodes, device=x.device).unsqueeze(0) < parts_num.unsqueeze(1)
		node_mask = node_mask.float()
		adj_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
		feat_mask = node_mask.unsqueeze(2)

		pos_weight = torch.tensor([12.0], device=x.device)
		loss_a_unreduced = F.binary_cross_entropy_with_logits(
			a_logits,
			adj,
			reduction='none',
			pos_weight=pos_weight
		)
		loss_a_masked = loss_a_unreduced * adj_mask
		loss_a = loss_a_masked.sum() / (adj_mask.sum() + 1e-8)

		# 2. Strata dla współrzędnych 3D (X)
		loss_x_unreduced = F.huber_loss(x_prime, x, reduction='none')
		loss_x_masked = loss_x_unreduced * feat_mask
		loss_x = loss_x_masked.sum() / (feat_mask.sum() + 1e-8)

		weight_a = self.hparams.get('weight_a', 1000.0)
		recon_loss = (weight_a * loss_a) + loss_x

		log_dict = {
			"loss_A": loss_a,
			"loss_X": loss_x,
		}

		return recon_loss, z, properties, log_dict