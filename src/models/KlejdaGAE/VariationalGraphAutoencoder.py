import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from typing import Dict, Any
from src.models.KlejdaGAE.Encoder import Encoder
from src.models.KlejdaGAE.DecoderA import DecoderA
from src.models.KlejdaGAE.DecoderX import DecoderX


class VariationalGraphAutoencoder(pl.LightningModule):
	def __init__(self, config: Dict[str, Any]):
		super().__init__()
		self.save_hyperparameters(config)

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

	def training_step(self, batch, batch_idx):
		x, adj = batch

		a_prime, x_prime, mu, logvar, _ = self(x, adj)

		loss_a = self.criterion(a_prime, adj)
		loss_x = self.criterion(x_prime, x)

		kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
		kl_loss = torch.mean(kl_loss)  # Uśredniamy karę dla całego batcha

		weight_a = self.hparams.get('weight_a', 1000.0)
		kl_weight = self.hparams.get('kl_weight', 1.0)  # Często beta < 1 pomaga zrekonstruować szczegóły

		total_loss = (weight_a * loss_a) + loss_x + (kl_weight * kl_loss)

		self.log("train/loss_total", total_loss, on_step=False, on_epoch=True)
		self.log("train/loss_A", loss_a, on_step=False, on_epoch=True)
		self.log("train/loss_X", loss_x, on_step=False, on_epoch=True)
		self.log("train/loss_KL", kl_loss, on_step=False, on_epoch=True)

		return total_loss

	def validation_step(self, batch, batch_idx):
		x, adj = batch
		a_prime, x_prime, mu, logvar, _ = self(x, adj)

		loss_a = self.criterion(a_prime, adj)
		loss_x = self.criterion(x_prime, x)

		kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
		kl_loss = torch.mean(kl_loss)

		weight_a = self.hparams.get('weight_a', 1000.0)
		kl_weight = self.hparams.get('kl_weight', 1.0)

		val_loss = (weight_a * loss_a) + loss_x + (kl_weight * kl_loss)
		self.log("val/loss_total", val_loss, on_step=False, on_epoch=True)

		return val_loss

	def configure_optimizers(self):
		return Adam(self.parameters(), lr=self.hparams.learning_rate)