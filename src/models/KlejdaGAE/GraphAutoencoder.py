import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from typing import Dict, Any
from src.models.KlejdaGAE.Encoder import Encoder
from src.models.KlejdaGAE.DecoderA import DecoderA
from src.models.KlejdaGAE.DecoderX import DecoderX

class GraphAutoencoder(pl.LightningModule):
	"""
	Pełny model Grafowego Autoenkodera (GAE) zaimplementowany w PyTorch Lightning.
	"""

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
			out_features=self.hparams.in_channels,  # np. 3 dla X, Y, Z
			max_nodes=self.hparams.max_nodes
		)

		# Standardowe kryterium MSE
		self.criterion = nn.MSELoss()

	def forward(self, x: torch.Tensor, adj: torch.Tensor):
		hidden_features = self.encoder_backbone(x, adj)
		z = self.fc_z(hidden_features)
		a_prime = self.decoder_a(z)
		x_prime = self.decoder_x(z, a_prime)

		return a_prime, x_prime, z

	def training_step(self, batch, batch_idx):
		x, adj = batch
		a_prime, x_prime, _ = self(x, adj)

		loss_a = self.criterion(a_prime, adj)
		loss_x = self.criterion(x_prime, x)

		weight_a = self.hparams.get('weight_a', 1000.0)
		total_loss = (weight_a * loss_a) + loss_x

		self.log("train/loss_total", total_loss, on_step=False, on_epoch=True, prog_bar=True)
		self.log("train/loss_A", loss_a, on_step=False, on_epoch=True, prog_bar=False)
		self.log("train/loss_X", loss_x, on_step=False, on_epoch=True, prog_bar=False)

		return total_loss

	def validation_step(self, batch, batch_idx):
		x, adj = batch
		a_prime, x_prime, _ = self(x, adj)

		loss_a = self.criterion(a_prime, adj)
		loss_x = self.criterion(x_prime, x)

		weight_a = self.hparams.get('weight_a', 1000.0)
		val_loss = (weight_a * loss_a) + loss_x

		self.log("val/loss_total", val_loss, on_step=False, on_epoch=True, prog_bar=True)
		return val_loss

	def configure_optimizers(self):
		optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
		return optimizer