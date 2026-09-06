import torch
from utils.FramsticksPostProcessor import FramsticksPostProcessor
from src.deap.native_operators import frams_evaluate
from src.deap.constraints import FITNESS_VALUE_INFEASIBLE_SOLUTION
import numpy as np

class AutoencoderEvaluator:
	def __init__(self, autoencoder,frams_lib, opt_criteria, config):
		self.autoencoder = autoencoder
		self.frams_evaluate = frams_evaluate
		self.frams_lib = frams_lib
		self.opt_criteria= opt_criteria
		self.config = config
		self.postProcessor = FramsticksPostProcessor(repair=False)

	def __call__(self, individual):
		latent_vector = torch.tensor([individual], dtype=torch.float32)

		# Dekodowanie wektora przestrzeni ukrytej
		with torch.no_grad():
			x_prime, a_prime = self.autoencoder.decode(latent_vector)

		x_prime.squeeze_()
		a_prime.squeeze_()

		# Konwersja macierzy do reprezentacji genotypowej
		is_valid_reconstruct, new_framsticks_genotype, _, _, flags = self.postProcessor.process(x_prime, a_prime)

		if type(new_framsticks_genotype) is not list:
			new_framsticks_genotype = [new_framsticks_genotype]

		# Ewaluacja genotypu w środowisku Framsticks
		fitness_tuple = self.frams_evaluate(self.frams_lib,self.opt_criteria,self.config, new_framsticks_genotype)

		# Kara za oddalenie się od miejsca początkowego
		distance_from_zero = np.linalg.norm(latent_vector)
		penalty = 0
		# if distance_from_zero > 15:
		# 	penalty = (distance_from_zero - 15) * 10


		is_valid_frams_evaluate = True
		fitness_tuple = list(fitness_tuple)
		for i in range(len(fitness_tuple)):
			if fitness_tuple[i] == FITNESS_VALUE_INFEASIBLE_SOLUTION:
				is_valid_frams_evaluate = False
				break
			fitness_tuple[i] -= penalty
		fitness_tuple = tuple(fitness_tuple)

		if type(new_framsticks_genotype) is list:
			new_framsticks_genotype = new_framsticks_genotype[0]

		return fitness_tuple, is_valid_reconstruct, is_valid_frams_evaluate, new_framsticks_genotype, flags