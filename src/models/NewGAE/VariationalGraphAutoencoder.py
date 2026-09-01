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

		self.apply(self._init_weights)

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
		x, adj, properties = batch
		parts_num = properties['parts_num']

		x_prime, a_logits, mu, logvar, z = self.forward(x, adj)

		# Maski
		node_mask = torch.arange(self.hparams.max_nodes, device=x.device).unsqueeze(0) < parts_num.unsqueeze(1)
		node_mask = node_mask.float()
		adj_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
		feat_mask = node_mask.unsqueeze(2)

		pos_weight = torch.tensor([self.hparams.pos_weight], device=x.device)

		loss_a_unreduced = F.binary_cross_entropy_with_logits(
			a_logits,
			adj,
			reduction='none',
			pos_weight=pos_weight
		)
		loss_a_masked = loss_a_unreduced * adj_mask
		loss_a = loss_a_masked.sum() / (adj_mask.sum() + 1e-6)

		loss_x_unreduced = F.huber_loss(x_prime, x, reduction='none')
		loss_x_masked = loss_x_unreduced * feat_mask
		loss_x = loss_x_masked.sum() / (feat_mask.sum() + 1e-6)

		# Dywergencja Kullbacka-Leiblera (kara za odchylenie z od rozkładu normalnego N(0,1))
		kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
		kl_loss = torch.mean(kl_loss)

		weight_a = self.hparams.get('weight_a', 500.0)
		kl_weight = self.hparams.get('kl_weight', 0.007)

		# Całkowity błąd
		recon_loss = (weight_a * loss_a) + loss_x + (kl_weight * kl_loss)

		with torch.no_grad():
			# Obliczanie dodatkowych metryk
			a_probs = torch.sigmoid(a_logits)
			# TODO: Jakiś threshold z configa tutaj?
			a_preds = (a_probs > 0.5).float()

			valid_mask = adj_mask.bool()
			preds_flat = a_preds[valid_mask]
			targets_flat = adj[valid_mask]

			# Zliczanie True Positives, True Negatives itp.
			TP = ((preds_flat == 1) & (targets_flat == 1)).sum().float()
			TN = ((preds_flat == 0) & (targets_flat == 0)).sum().float()
			FP = ((preds_flat == 1) & (targets_flat == 0)).sum().float()
			FN = ((preds_flat == 0) & (targets_flat == 1)).sum().float()

			eps = 1e-8

			recall = TP / (TP + FN + eps)
			specificity = TN / (TN + FP + eps)
			precision = TP / (TP + FP + eps)
			g_mean = torch.sqrt(recall * specificity)
		log_dict = {
			"loss_A": loss_a,
			"loss_X": loss_x,
			"loss_KL": kl_loss,
			"metric_A_recall": recall,
			"metric_A_precision": precision,
			"metric_A_g_mean": g_mean,
			"metric_A_fp_ratio": FP / (FP + TN + eps)
		}

		return recon_loss, z, properties, log_dict