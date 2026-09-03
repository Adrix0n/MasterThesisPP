import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from src.models.NewGAE.Encoder import Encoder
from src.models.NewGAE.DecoderA import DecoderA
from src.models.NewGAE.DecoderX import DecoderX
from src.models.BaseGraphAutoEncoder import BaseGraphAutoEncoder


class VariationalGraphAutoencoder(BaseGraphAutoEncoder):

	def __init__(self, config: Dict[str, Any], frams_module):
		super().__init__(config, frams_module)

		# Inicjalizacja Enkodera
		self.encoder_backbone = Encoder(
			in_channels=self.hparams.in_channels,
			conv_channels=self.hparams.encoder_conv_channels,
			dense_features=self.hparams.encoder_dense_features,
			max_nodes=self.hparams.max_nodes,
			use_mlp=self.hparams.encoder_use_mlp,
			config=config
		)

		# Warstwy ukryte wariacyjnego autoenkodera
		self.fc_mu = nn.Linear(int(self.encoder_backbone.output_dim), self.hparams.latent_dim)
		self.fc_logvar = nn.Linear(int(self.encoder_backbone.output_dim), self.hparams.latent_dim)

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

	def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
		if self.training:
			std = torch.exp(0.5 * logvar)
			eps = torch.randn_like(std)
			return mu + eps * std
		else:
			# Podczas ewaluacji chcemy deterministycznych wyników (tylko średnia)
			return mu

	def forward(self, x: torch.Tensor, adj: torch.Tensor):
		hidden_features = self.encoder_backbone(x, adj)

		mu = self.fc_mu(hidden_features)
		logvar = self.fc_logvar(hidden_features)

		z = self.reparameterize(mu, logvar)

		# Dekoder A zwraca surowe logity
		a_logits = self.decoder_a(z)

		# Dekoder X potrzebuje prawdopodobieństw [0, 1]
		a_probs = torch.sigmoid(a_logits)
		x_prime = self.decoder_x(z, a_probs)

		# Zwracamy a_logits dla stabilnej funkcji straty
		return x_prime, a_logits, mu, logvar, z

	def encode(self, x: torch.Tensor, adj: torch.Tensor):
		hidden_features = self.encoder_backbone(x, adj)
		mu = self.fc_mu(hidden_features)
		logvar = self.fc_logvar(hidden_features)

		z = self.reparameterize(mu, logvar)
		return z

	def decode(self, z: torch.Tensor):
		a_logits = self.decoder_a(z)
		a_probs = torch.sigmoid(a_logits)
		x_prime = self.decoder_x(z, a_probs)

		return x_prime, a_probs

	def compute_reconstruction_loss(self, batch):
		# Pobranie wartości z batcha
		x, adj, properties = batch
		parts_num = properties['parts_num']

		# Utworzenie masek
		node_mask, adj_mask, feat_mask = self.create_masks(parts_num, device=x.device)

		# Przejście przez autoenkoder
		x_prime, a_logits, mu, logvar, z = self.forward(x, adj)

		# Strata A
		pos_weight = torch.tensor([self.hparams.pos_weight], device=x.device)
		loss_a_unreduced = F.binary_cross_entropy_with_logits(
			a_logits,
			adj,
			reduction='none',
			pos_weight=pos_weight
		)
		loss_a_masked = loss_a_unreduced * adj_mask
		loss_a = loss_a_masked.sum() / (adj_mask.sum() + 1e-6)

		# Strata X
		loss_x_unreduced = F.huber_loss(x_prime, x, reduction='none')
		loss_x_masked = loss_x_unreduced * feat_mask
		num_features = x.size(2)
		loss_x = loss_x_masked.sum() / (feat_mask.sum() * num_features + 1e-6)

		# Strata KL
		kl_unreduced = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
		kl_masked = kl_unreduced * node_mask.unsqueeze(-1)
		latent_dim = mu.size(-1)
		kl_loss = kl_masked.sum() / (node_mask.sum() * latent_dim + 1e-6)


		weight_a = self.hparams.weight_a
		kl_weight = self.hparams.kl_weight

		# Całkowity błąd
		recon_loss = (weight_a * loss_a) + loss_x + (kl_weight * kl_loss)

		# Dodatkowy, surowy błąd pomijający wagi i KL (Bo ono koniecznie musi być skalowane, ze względu na swoje olbrzymie wartości)
		with torch.no_grad():
			recon_loss_raw = loss_a + loss_x

		# Dodatkowe metryki
		metric_dict_a = self.compute_adj_metrics(torch.sigmoid(a_logits),adj,adj_mask,self.hparams.joint_threshold)
		log_dict = {
			"loss_A": loss_a,
			"loss_X": loss_x,
			"loss_KL": kl_loss,
			"recon_loss_raw": recon_loss_raw,
			**metric_dict_a,
		}

		return recon_loss, z, properties, log_dict