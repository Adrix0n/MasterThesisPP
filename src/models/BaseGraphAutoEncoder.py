import pytorch_lightning as pl
from typing import Dict, Any, Tuple
import torch.optim as optim
import torch
import torch.nn as nn
import os
import sys
current_dir = os.getcwd()
framspy_path = os.path.abspath(os.path.join(current_dir, '..', 'external', 'framspy'))
if framspy_path not in sys.path:
	sys.path.insert(0, framspy_path)
from dissimilarity.density_distribution import DensityDistribution
class BaseGraphAutoEncoder(pl.LightningModule):
	def __init__(self, config: Dict[str, Any], frams_module):
		super().__init__()
		self.save_hyperparameters(config)
		self.density_distribution = DensityDistribution(
			frams_module = frams_module,
			density = 10,			# default
			resolution = 8,			# default
			reduce_empty = True,	# default
			frequency = False,		# default
			metric = 'emd',			# default
			fixedZaxis = False,		# default
			verbose = False 		# default
		)

	# Korelacja Pearsona określa poziom liniowości zależności (zakres -1.0 ... 1.0)
	def _pearson_correlation(self, x: torch.Tensor, y: torch.Tensor):
		xm = x - torch.mean(x)
		ym = y - torch.mean(y)

		var_x = torch.sum(xm ** 2)
		var_y = torch.sum(ym ** 2)

		r_num = torch.sum(xm * ym)
		r_den = torch.sqrt(torch.clamp(var_x * var_y, min=1e-8))
		return r_num / r_den

	# Korelacja Spearmana określa monotoniczność zależności (zakres -1.0 ... 1.0)
	# Podstawowy Spearman jest nieróżniczkowalny,
	def _spearman_correlation(self, x: torch.Tensor, y: torch.Tensor):
		with torch.no_grad():
			rank_y = y.argsort(dim=-1).argsort(dim=-1).float()

			x_detached = x.detach()
			x_sorted, _ = torch.sort(x_detached, dim=-1)

			idx = torch.searchsorted(x_sorted, x_detached)
			idx_left = torch.clamp(idx - 1, min=0, max=x.size(-1) - 2)
			idx_right = idx_left + 1

		val_left = torch.gather(x_sorted, -1, idx_left)
		val_right = torch.gather(x_sorted, -1, idx_right)

		diff = val_right - val_left

		fraction = torch.where(
			diff > 1e-6,
			(x - val_left) / (diff + 1e-8),  # +1e-8 chroni przed problemami precyzji float
			torch.zeros_like(x)
		)

		rank_x = idx_left.float() + fraction

		return self._pearson_correlation(rank_x, rank_y)

	# Inicjalizator wag
	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
			if m.bias is not None:
				nn.init.constant_(m.bias, 0.0)

		elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
			nn.init.constant_(m.weight, 1.0)
			nn.init.constant_(m.bias, 0.0)



	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
		scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: ((epoch+1)/10.0) ** 0.7)
		return {
			"optimizer": optimizer,
			"lr_scheduler": {
				"scheduler": scheduler,
				"interval": "epoch",
				"monitor": "val/loss_total",
			},
		}

	def compute_reconstruction_loss(self, batch):
		raise NotImplementedError("Klasa dziedzicząca musi implementować metodę 'compute_reconstruction_loss'")

	def compute_locality_loss(self, z: torch.Tensor, properties: Dict[str, any]) -> Tuple[
		torch.Tensor, torch.Tensor]:
		if self.hparams.get('locality_loss_method') is None:
			return torch.tensor(0.0, device=z.device), torch.tensor(0.0, device=z.device)

		batch_size = z.size(0)
		if batch_size <= 1:
			return torch.tensor(0.0, device=z.device), torch.tensor(0.0, device=z.device)

		loss_type = self.hparams.locality_loss_type
		loss_method = self.hparams.locality_loss_method
		lambda_val = self.hparams.locality_loss_lambda_val

		z_dist = torch.cdist(z, z, p=2.0)

		# Dla similarity należy skorzystać z biblioteki Framsticks, która pozwala wyliczyć macierz dissimilarity
		if loss_type == 'similarity':
			genotypes = properties['genotype']
			dissimilarity_matrix = self.density_distribution.getDissimilarityMatrix(genotypes)
			prop_dist = torch.tensor(dissimilarity_matrix, dtype=torch.float32, device=z.device)
		else:
			prop = properties[loss_type].squeeze()
			prop_unsq = prop.unsqueeze(1)
			prop_dist = torch.cdist(prop_unsq, prop_unsq, p=1.0)

		# Pobranie unikalnych par (górny trójkąt bez przekątnej)
		row, col = torch.triu_indices(batch_size, batch_size, offset=1)
		z_dist_flat = z_dist[row, col]
		prop_dist_flat = prop_dist[row, col]

		# Wybór metody i obliczanie straty oraz korelacji (do logów)
		if loss_method == 'pearson':
			corr = self._pearson_correlation(z_dist_flat, prop_dist_flat)
			loss = lambda_val * (1 - corr)

		elif loss_method == 'spearman':
			corr = self._spearman_correlation(z_dist_flat, prop_dist_flat)
			loss = lambda_val * (1 - corr)
		else:
			raise ValueError(f"Nieznana metoda: {loss_method}")

		return loss, corr

	def train_val_step(self, batch, step_type):
		recon_loss, z, properties, log_dict = self.compute_reconstruction_loss(batch)

		locality_loss, correlation = self.compute_locality_loss(z, properties)

		total_loss = recon_loss + locality_loss

		self.log(f"{step_type}/loss_total", total_loss, on_step=False, on_epoch=True, prog_bar=True)
		self.log(f"{step_type}/locality_correlation", correlation, on_step=False, on_epoch=True)
		self.log(f"{step_type}/loss_locality", locality_loss, on_step=False, on_epoch=True)

		for key, value in log_dict.items():
			self.log(f"{step_type}/{key}", value, on_step=False, on_epoch=True)

		return total_loss

	def training_step(self, batch, batch_idx):
		return self.train_val_step(batch, step_type='train')

	def validation_step(self, batch, batch_idx):
		return self.train_val_step(batch, step_type='val')

	def forward(self, *args, **kwargs):
		raise NotImplementedError("Klasa dziedzicząca musi implementować metodę 'forward'")