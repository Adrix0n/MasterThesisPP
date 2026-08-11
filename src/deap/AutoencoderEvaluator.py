import torch
from utils.FramsticksPostProcessor import FramsticksPostProcessor
from src.deap.native_operators import frams_evaluate


class AutoencoderEvaluator:
	def __init__(self, autoencoder,frams_lib, opt_criteria, config):
		self.autoencoder = autoencoder
		self.frams_evaluate = frams_evaluate
		self.frams_lib = frams_lib
		self.opt_criteria= opt_criteria
		self.config = config
		self.postProcessor = FramsticksPostProcessor()

	def __call__(self, individual):
		latent_vector = torch.tensor([individual], dtype=torch.double)

		# Dekodowanie wektora przestrzeni ukrytej
		with torch.no_grad():
			x_prime, a_prime = self.autoencoder.decode(latent_vector)

		x_prime.squeeze_()
		a_prime.squeeze_()

		# Konwersja macierzy do reprezentacji genotypowej
		_, new_framsticks_genotype, _, _ = self.postProcessor.process(x_prime, a_prime)

		if type(new_framsticks_genotype) is not list:
			new_framsticks_genotype = [new_framsticks_genotype]

		# Ewaluacja genotypu w środowisku Framsticks
		fitness_tuple = self.frams_evaluate(self.frams_lib,self.opt_criteria,self.config, new_framsticks_genotype)

		return fitness_tuple