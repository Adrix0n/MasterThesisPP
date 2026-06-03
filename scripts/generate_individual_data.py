import yaml
import sys
import os

# Dodanie ścieżki, aby działało środowisko
framspy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'external', 'framspy'))

if framspy_path not in sys.path:
	sys.path.insert(0, framspy_path)
from FramsticksLib import FramsticksLib
from deap import tools, algorithms
import time
import numpy as np
from src.deap.deap_setup import prepare_native_toolbox
from src.deap.constraints import is_feasible_fitness_criteria
import json
import random

def accumulate_samples(population, generation, fraction):
	result = []
	sample_size = max(1, int(len(population) * fraction))
	sampled_individuals = random.sample(population, sample_size)
	for ind in sampled_individuals:
		result.append({
			"generation": generation,
			"fitness": ind.fitness.values,
			"genotype": str(ind[0]),
		})
	return result

def main():
	with open("../configs/generate_individual_data_config.yaml") as f:
		config = yaml.safe_load(f)

	frams_lib = FramsticksLib(config['frams_path'], config['frams_lib'], config['sim_file'])

	toolbox = prepare_native_toolbox(frams_lib, config)
	pop = toolbox.population(n=config['pop_size'])

	stats = tools.Statistics(lambda ind: ind.fitness.values)
	filter_feasible = lambda func, criteria: func(list(filter(is_feasible_fitness_criteria, criteria)))
	stats.register("min", lambda fit: filter_feasible(np.min, fit))
	stats.register("avg", lambda fit: filter_feasible(np.mean, fit))
	stats.register("max", lambda fit: filter_feasible(np.max, fit))

	logbook = tools.Logbook()
	logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

	hof = tools.HallOfFame(config['hof_size'])

	for gen in range(1, config['generations'] + 1):
		pop, _ = algorithms.eaSimple(pop, toolbox,
									 cxpb=config['p_xov'],
									 mutpb=config['p_mut'],
									 ngen=1,
									 stats=None,
									 halloffame=hof,
									 verbose=False)

		samples = accumulate_samples( pop, generation=gen, fraction=config['sampling_fraction'])

		record = stats.compile(pop) if stats else {}
		logbook.record(gen=gen, nevals=len(pop), **record)
		print(logbook.stream)
		with open(config["result_filepath"], "a") as f:
			for sample in samples:
				f.write(json.dumps(sample) + "\n")

if __name__ == "__main__":
	main()
