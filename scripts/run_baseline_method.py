from src.deap.save_and_load_results import save_genotypes_json
import sys
import os

framspy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'external', 'framspy'))

if framspy_path not in sys.path:
	sys.path.insert(0, framspy_path)

from FramsticksLib import FramsticksLib
from deap import tools, algorithms
import time
import numpy as np
import yaml
from src.deap.deap_setup import prepare_native_toolbox
from src.deap.constraints import is_feasible_fitness_criteria
from src.deap.save_and_load_results import save_genotypes_json


def main():
	with open("../configs/run_baseline_method.yaml", 'r') as f:
		config = yaml.safe_load(f)

	frams_lib = FramsticksLib(config['frams_path'], config['frams_lib'], config['sim_file'])

	toolbox = prepare_native_toolbox(frams_lib, config)
	pop = toolbox.population(n=config['pop_size'])
	hof = tools.HallOfFame(config['hof_size'])

	stats = tools.Statistics(lambda ind: ind.fitness.values)
	filter_feasible = lambda func, criteria: func(list(filter(is_feasible_fitness_criteria, criteria)))
	stats.register("min", lambda fit: filter_feasible(np.min, fit))
	stats.register("avg", lambda fit: filter_feasible(np.mean, fit))
	stats.register("max", lambda fit: filter_feasible(np.max, fit))

	pop, log = algorithms.eaSimple(
		pop, toolbox,
		cxpb=config['p_xov'], mutpb=config['p_mut'],
		ngen=config['generations'], stats=stats, halloffame=hof, verbose=True
	)

	print(f"\nNajlepszy fitness w HoF: {hof[0].fitness.values[0]}")
	save_genotypes_json(config["result_filepath"], hof)


if __name__ == "__main__":
	main()
