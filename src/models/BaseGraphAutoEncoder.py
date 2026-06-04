import pytorch_lightning as pl
from typing import Dict, Any, Tuple
import torch.optim as optim
import torch
class BaseGraphAutoEncoder(pl.LightningModule):
	def __init__(self,config: Dict[str, Any]):
		super().__init__()
		self.save_hyperparameters(config)

	def _pearson_correlation(self, x: torch.Tensor, y: torch.Tensor):
		xm = x - torch.mean(x)
		ym = y - torch.mean(y)
		r_num = torch.sum(xm * ym)
		r_den = torch.clamp(torch.norm(xm) * torch.norm(ym), min=1e-8)
		return r_num / r_den

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5)

		return {
			"optimizer": optimizer,
			"lr_scheduler": {
				"scheduler": scheduler,
				"monitor": "val/loss_total",
			},
		}
	def compute_reconstruction_loss(self, batch):
		raise NotImplementedError("Klasa dziedzicząca musi implementować metodę 'compute_reconstruction_loss'")

	def compute_locality_loss(self, z: torch.Tensor , properties: Dict[str,torch.Tensor]):
		if self.hparams.locality_loss_method is None:
			return torch.tensor(0.0, device=z.device, requires_grad=True), torch.tensor(0.0, device=z.device)
		batch_size = z.size(0)
		if batch_size <= 1:
			return torch.tensor(0.0, device=z.device, requires_grad=True), torch.tensor(0.0, device=z.device)

		loss_type =self.hparams.locality_loss_type
		loss_method = self.hparams.locality_loss_method
		lambda_val = self.hparams.locality_loss_lambda_val

		prop = properties[loss_type].squeeze()

		distances = torch.cdist(z,z,p=2.0)
		mean_distances = distances.sum(dim=1) / (batch_size - 1)

		if loss_method == 'pearson':
			corr = self._pearson_correlation(mean_distances, prop)
			loss = lambda_val * (1-corr)
		else:
			raise ValueError

		return corr, loss



	def train_val_step(self, batch, step_type):
		recon_loss, z, properties, log_dict = self.compute_reconstruction_loss(batch)

		locality_loss, correlation = self.compute_locality_loss(z, properties)

		total_loss = recon_loss + locality_loss / 2

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