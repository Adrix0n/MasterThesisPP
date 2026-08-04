import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from typing import Dict, Any
from src.models.KlejdaGAE.Encoder import Encoder
from src.models.KlejdaGAE.DecoderA import DecoderA
from src.models.KlejdaGAE.DecoderX import DecoderX
from src.models.BaseGraphAutoEncoder import BaseGraphAutoEncoder


class VariationalGraphAutoencoder(BaseGraphAutoEncoder):
	def __init__(self, config: Dict[str, Any], frams_module):
		super().__init__(config, frams_module)

		# Inicjalizacja Enkodera
		self.encoder_backbone = Encoder(
			in_channels=self.hparams.in_channels,
			conv_channels=self.hparams.encoder_conv_channels,
			dense_features=self.hparams.encoder_dense_features,
			max_nodes=self.hparams.max_nodes
		)

		# Warstwy ukryte wariacyjnego autoenkodera
		self.fc_mu = nn.Linear(self.encoder_backbone.output_dim, self.hparams.latent_dim)
		self.fc_logvar = nn.Linear(self.encoder_backbone.output_dim, self.hparams.latent_dim)

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

	def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
		if self.training:
			std = torch.exp(0.5 * logvar)
			eps = torch.randn_like(std)
			return mu + eps * std
		else:
			# Podczas ewaluacji chcemy deterministycznych wyników
			return mu

	def forward(self, x: torch.Tensor, adj: torch.Tensor):
		hidden_features = self.encoder_backbone(x, adj)

		mu = self.fc_mu(hidden_features)
		logvar = self.fc_logvar(hidden_features)

		z = self.reparameterize(mu, logvar)

		a_prime = self.decoder_a(z)
		x_prime = self.decoder_x(z, a_prime)

		return a_prime, x_prime, mu, logvar, z

	def compute_reconstruction_loss(self, batch):
		# Rozpakowanie batcha
		x, adj, properties = batch

		# Przepływ przez ten konkretny model
		a_prime, x_prime, mu, logvar, z = self.forward(x, adj)

		# Liczenie specyficznych strat
		loss_a = self.criterion(a_prime, adj)
		loss_x = self.criterion(x_prime, x)
		weight_a = self.hparams.get('weight_a', 1000.0)

		# Dywergencja Kullbacka_Liblera
		kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
		kl_loss = torch.mean(kl_loss)  # Uśredniamy karę dla całego batcha
		kl_weight = self.hparams.get('kl_weight', 1.0)  # Często beta < 1 pomaga zrekonstruować szczegóły

		recon_loss = (weight_a * loss_a) + loss_x + (kl_weight * kl_loss)

		log_dict = {
			"loss_A": loss_a,
			"loss_X": loss_x,
			"loss_KL": kl_loss,
		}

		return recon_loss, z, properties, log_dict

	def encode(self, x: torch.Tensor, adj: torch.Tensor):
		hidden_features = self.encoder_backbone(x, adj)
		mu = self.fc_mu(hidden_features)
		logvar = self.fc_logvar(hidden_features)

		z = self.reparameterize(mu, logvar)
		return z

	def decode(self, z: torch.Tensor):
		a_prime = self.decoder_a(z)
		x_prime = self.decoder_x(z, a_prime)

		return x_prime, a_prime