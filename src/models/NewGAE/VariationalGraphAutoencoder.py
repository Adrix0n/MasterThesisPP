import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from src.models.NewGAE.Encoder import Encoder
from src.models.NewGAE.DecoderA import DecoderA
from src.models.NewGAE.DecoderX import DecoderX
from src.models.BaseGraphAutoEncoder import BaseGraphAutoEncoder


class FocalLoss(nn.Module):
	"""
	Focal Loss for sparse binary adjacency matrices.
	Reduces loss contribution from easy negatives (abundant zeros)
	and focuses on hard positives (rare edges).
	"""
	def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
		super().__init__()
		self.alpha = alpha
		self.gamma = gamma

	def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		bce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
		pt = torch.exp(-bce_loss)  # probability of correct prediction
		focal_weight = self.alpha * (1 - pt) ** self.gamma
		focal_loss = focal_weight * bce_loss
		return focal_loss.mean()


class VariationalGraphAutoencoder(BaseGraphAutoEncoder):

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

		# Warstwy ukryte wariacyjnego autoenkodera
		self.fc_mu = nn.Linear(self.encoder_backbone.output_dim, self.hparams.latent_dim)
		self.fc_logvar = nn.Linear(self.encoder_backbone.output_dim, self.hparams.latent_dim)

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

		# Focal Loss dla rzadkiej macierzy sąsiedztwa
		self.criterion_a = FocalLoss(alpha=0.25, gamma=2.0)
		self.criterion_x = nn.HuberLoss()

		self.apply(self._init_weights)

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
		x, adj, properties = batch
		parts_num = properties['parts_num']  # (B,) — liczba rzeczywistych węzłów

		a_prime, x_prime, mu, logvar, z = self.forward(x, adj)

		# Maska węzłów: (B, max_nodes)
		node_mask = torch.arange(self.hparams.max_nodes, device=x.device).unsqueeze(0) < parts_num.unsqueeze(1)
		node_mask = node_mask.float()

		# Maska sąsiedztwa: (B, max_nodes, max_nodes)
		adj_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)

		# Maska cech: (B, max_nodes, 1) → broadcast do (B, max_nodes, num_features)
		feat_mask = node_mask.unsqueeze(2)

		# Straty z maskowaniem
		loss_a = self.criterion_a(a_prime * adj_mask, adj * adj_mask)
		loss_x = self.criterion_x(x_prime * feat_mask, x * feat_mask)

		# Normalizacja przez liczbę rzeczywistych elementów
		loss_a = loss_a * (adj_mask.numel() / (adj_mask.sum() + 1e-8))
		loss_x = loss_x * (feat_mask.numel() / (feat_mask.sum() + 1e-8))

		weight_a = self.hparams.get('weight_a', 1000.0)

		# Dywergencja Kullbacka_Liblera
		kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
		kl_loss = torch.mean(kl_loss)  # Uśredniamy karę dla całego batcha
		kl_weight = self.hparams.get('kl_weight', 1.0)

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

		return a_prime, x_prime