import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from typing import Dict, Any
from src.models.KlejdaGAE.KlejdaEncoder import KlejdaEncoder
from src.models.KlejdaGAE.KlejdaDecoderA import KlejdaDecoderA
from src.models.KlejdaGAE.KlejdaDecoderX import KlejdaDecoderX
from src.models.BaseGraphAutoEncoder import BaseGraphAutoEncoder
class KlejdaGraphAutoencoder(BaseGraphAutoEncoder):
	"""
	Pełny model Grafowego Autoenkodera (GAE) zaimplementowany w PyTorch Lightning.
	"""

	def __init__(self, config: Dict[str, Any], frams_module):
		super().__init__(config, frams_module)

		# Inicjalizacja Enkodera
		self.encoder_backbone = KlejdaEncoder(
			in_channels=self.hparams.in_channels,
			conv_channels=self.hparams.encoder_conv_channels,
			dense_features=self.hparams.encoder_dense_features,
			max_nodes=self.hparams.max_nodes
		)

		# Warstwa rzutująca do przestrzeni ukrytej Z
		self.fc_z = nn.Linear(int(self.encoder_backbone.output_dim), self.hparams.latent_dim)

		# Inicjalizacja Dekodera A
		self.decoder_a = KlejdaDecoderA(
			latent_dim=self.hparams.latent_dim,
			hidden_dims=self.hparams.decoder_a_hidden_dims,
			max_nodes=self.hparams.max_nodes
		)

		# Inicjalizacja Dekodera X
		self.decoder_x = KlejdaDecoderX(
			latent_dim=self.hparams.latent_dim,
			conv_channels=self.hparams.decoder_x_conv_channels,
			dense_features=self.hparams.decoder_x_dense_features,
			out_features=self.hparams.in_channels,
			max_nodes=self.hparams.max_nodes
		)

		self.criterion = nn.MSELoss()



	def forward(self, x: torch.Tensor, adj: torch.Tensor):
		hidden_features = self.encoder_backbone(x, adj)
		z = self.fc_z(hidden_features)
		a_prime = self.decoder_a(z)
		x_prime = self.decoder_x(z, a_prime)

		return x_prime, a_prime, z

	def compute_reconstruction_loss(self, batch):
		# Rozpakowanie batcha
		x, adj, properties = batch
		parts_num = properties['parts_num']

		# Utworzenie masek
		node_mask, adj_mask, feat_mask = self.create_masks(parts_num, device=x.device)

		# Przepływ przez ten konkretny model
		x_prime, a_prime, z = self.forward(x, adj)

		# Strata A
		loss_a_unreduced = self.criterion(a_prime, adj)
		loss_a_masked = loss_a_unreduced * adj_mask
		loss_a = loss_a_masked.sum() / (adj_mask.sum() + 1e-6)

		# Strata X
		loss_x_unreduced = self.criterion(x_prime, x)
		loss_x_masked = loss_x_unreduced * feat_mask
		num_features = x.size(2)
		loss_x = loss_x_masked.sum() / (feat_mask.sum() * num_features + 1e-6)

		weight_a = self.hparams.weight_a
		# Całkowity błąd
		recon_loss = (weight_a * loss_a) + loss_x

		# Dodatkowy, surowy błąd pomijający wagi
		with torch.no_grad():
			recon_loss_raw = loss_a + loss_x

		# Dodatkowe metryki
		metric_dict_a = self.compute_adj_metrics(a_prime, adj, adj_mask, self.hparams.joint_threshold)
		# Słownik z dodatkowymi wartościami do zalogowania
		log_dict = {
			"loss_A": loss_a,
			"loss_X": loss_x,
			"recon_loss_raw":recon_loss_raw,
			**metric_dict_a
		}

		# Zgodnie z kontraktem, zwracamy 4 rzeczy do klasy bazowej
		return recon_loss, z, properties, log_dict

	def encode(self, x: torch.Tensor, adj: torch.Tensor):
		hidden_features = self.encoder_backbone(x, adj)
		z = self.fc_z(hidden_features)
		return z

	def decode(self, z: torch.Tensor):
		a_prime = self.decoder_a(z)
		x_prime = self.decoder_x(z, a_prime)

		return x_prime, a_prime