import math
from pytorch_lightning.callbacks import GradientAccumulationScheduler


class QuadraticGradientAccumulationScheduler(GradientAccumulationScheduler):
	def __init__(
			self,
			start_multiplier: int,
			end_multiplier: int,
			step: int,
			epochs: int
	):
		scheduling = {}
		for epoch in range(0, epochs + 1, step):
			fraction = epoch / epochs

			exact_value = start_multiplier + (end_multiplier - start_multiplier) * (fraction ** 2)


			multiplier = int(round(exact_value))

			if not scheduling or multiplier > list(scheduling.values())[-1]:
				scheduling[epoch] = multiplier

		if epochs not in scheduling and list(scheduling.values())[-1] < end_multiplier:
			scheduling[epochs] = end_multiplier

		self.generated_scheduling = scheduling
		super().__init__(scheduling=scheduling)
