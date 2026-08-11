import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from typing import Dict, Any
from src.models.KlejdaGAE.Encoder import Encoder
from src.models.KlejdaGAE.DecoderA import DecoderA
from src.models.KlejdaGAE.DecoderX import DecoderX
from src.models.BaseGraphAutoEncoder import BaseGraphAutoEncoder
class GraphAutoencoder(BaseGraphAutoEncoder):
	"""
	Pełny model Grafowego Autoenkodera (GAE) zaimplementowany w PyTorch Lightning.
	"""

	def __init__(self, config: Dict[str, Any], frams_module):
		super().__init__(config, frams_module)

		# Inicjalizacja Enkodera
		self.encoder_backbone = Encoder(
			in_channels=self.hparams.in_channels,
			conv_channels=self.hparams.encoder_conv_channels,
			dense_features=self.hparams.encoder_dense_features,
			max_nodes=self.hparams.max_nodes
		)

		# Warstwa rzutująca do przestrzeni ukrytej Z
		self.fc_z = nn.Linear(self.encoder_backbone.output_dim, self.hparams.latent_dim)

		# Inicjalizacja Dekodera A
		self.decoder_a = DecoderA(
			latent_dim=self.hparams.latent_dim,
			hidden_dims=self.hparams.decoder_a_hidden_dims,
			max_nodes=self.hparams.max_nodes
		)

		# Inicjalizacja Dekodera X
		self.decoder_x = DecoderX(
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

		return a_prime, x_prime, z

	def compute_reconstruction_loss(self, batch):
		# Rozpakowanie batcha
		x, adj, properties = batch

		# Przepływ przez ten konkretny model
		a_prime, x_prime, z = self.forward(x, adj)

		# Liczenie specyficznych strat
		loss_a = self.criterion(a_prime, adj)
		loss_x = self.criterion(x_prime, x)
		weight_a = self.hparams.get('weight_a', 1000.0)

		recon_loss = (weight_a * loss_a) + loss_x

		valid_percentage = None
		if self.hparams.count_valid:
			valid_percentage = self.calc_valid_perc(x_prime,a_prime)


		# Słownik z dodatkowymi wartościami do zalogowania
		log_dict = {
			"loss_A": loss_a,
			"loss_X": loss_x,
			"valid_percentage": valid_percentage
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