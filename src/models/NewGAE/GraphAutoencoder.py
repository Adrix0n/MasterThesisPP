import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
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
			use_mlp=True,
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

		self.criterion_a = nn.BCEWithLogitsLoss()
		self.criterion_x = nn.HuberLoss()

		self.apply(self._init_weights)


	def forward(self, x: torch.Tensor, adj: torch.Tensor):
		hidden_features = self.encoder_backbone(x, adj)
		z = self.fc_z(hidden_features)
		a_prime = self.decoder_a(z)
		x_prime = self.decoder_x(z)

		return a_prime, x_prime, z

	def compute_reconstruction_loss(self, batch):
		x, adj, properties = batch

		a_prime, x_prime, z = self.forward(x, adj)

		loss_a = self.criterion_a(a_prime, adj)
		loss_x = self.criterion_x(x_prime, x)
		weight_a = self.hparams.get('weight_a', 1000.0)

		recon_loss = (weight_a * loss_a) + loss_x

		# Słownik z dodatkowymi wartościami do zalogowania
		log_dict = {
			"loss_A": loss_a,
			"loss_X": loss_x,
		}

		return recon_loss, z, properties, log_dict

	def encode(self, x: torch.Tensor, adj: torch.Tensor):
		hidden_features = self.encoder_backbone(x, adj)
		z = self.fc_z(hidden_features)
		return z

	def decode(self, z: torch.Tensor):
		a_prime = self.decoder_a(z)
		x_prime = self.decoder_x(z)

		return x_prime, a_prime