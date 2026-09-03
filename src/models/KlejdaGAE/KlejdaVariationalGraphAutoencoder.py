import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from typing import Dict, Any
from src.models.KlejdaGAE.KlejdaEncoder import KlejdaEncoder
from src.models.KlejdaGAE.KlejdaDecoderA import KlejdaDecoderA
from src.models.KlejdaGAE.KlejdaDecoderX import KlejdaDecoderX
from src.models.BaseGraphAutoEncoder import BaseGraphAutoEncoder


class KlejdaVariationalGraphAutoencoder(BaseGraphAutoEncoder):
	def __init__(self, config: Dict[str, Any], frams_module):
		super().__init__(config, frams_module)

		# Inicjalizacja Enkodera
		self.encoder_backbone = KlejdaEncoder(
			in_channels=self.hparams.in_channels,
			conv_channels=self.hparams.encoder_conv_channels,
			dense_features=self.hparams.encoder_dense_features,
			max_nodes=self.hparams.max_nodes
		)

		# Warstwy ukryte wariacyjnego autoenkodera
		self.fc_mu = nn.Linear(int(self.encoder_backbone.output_dim), self.hparams.latent_dim)
		self.fc_logvar = nn.Linear(int(self.encoder_backbone.output_dim), self.hparams.latent_dim)

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

		return x_prime, a_prime, mu, logvar, z

	def compute_reconstruction_loss(self, batch):
		# Rozpakowanie batcha
		x, adj, properties = batch
		parts_num = properties['parts_num']

		# Utworzenie masek
		node_mask, adj_mask, feat_mask = self.create_masks(parts_num, device=x.device)

		# Przepływ przez ten konkretny model
		x_prime, a_prime, mu, logvar, z = self.forward(x, adj)

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
		kl_weight = self.hparams.kl_weight

		# Strata KL
		kl_unreduced = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
		kl_masked = kl_unreduced * node_mask.unsqueeze(-1)
		latent_dim = mu.size(-1)
		kl_loss = kl_masked.sum() / (node_mask.sum() * latent_dim + 1e-6)


		recon_loss = (weight_a * loss_a) + loss_x + (kl_weight * kl_loss)
		# Dodatkowy, surowy błąd pomijający wagi i KL (Bo ono koniecznie musi być skalowane, ze względu na swoje olbrzymie wartości)
		with torch.no_grad():
			recon_loss_raw = loss_a + loss_x

		metric_dict_a = self.compute_adj_metrics(a_prime,adj,adj_mask,self.hparams.joint_threshold)
		log_dict = {
			"loss_A": loss_a,
			"loss_X": loss_x,
			"loss_KL": kl_loss,
			"recon_loss_raw": recon_loss_raw,
			**metric_dict_a,
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