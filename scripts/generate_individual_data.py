import yaml
import sys
import os
import json
import numpy as np
import random
import multiprocessing
from deap import tools, algorithms
import secrets

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
framspy_path = os.path.join(project_path, 'external', 'framspy')

sys.path.insert(0, project_path)
sys.path.insert(0, framspy_path)

from FramsticksLib import FramsticksLib
from src.deap.deap_setup import prepare_native_toolbox
from src.deap.constraints import is_feasible_fitness_criteria


def run_single_experiment(args):
	_, config = args

	frams_lib = FramsticksLib(
		config['frams_path'],
		config['frams_lib'],
		config['sim_file']
	)

	toolbox = prepare_native_toolbox(frams_lib, config)

	orig_filepath = config["result_filepath"]
	base_name, ext = os.path.splitext(orig_filepath)

	evolution_counter = 1

	while True:
		run_id = secrets.token_hex(16)

		random.seed(run_id)
		np.random.seed(int.from_bytes(os.urandom(4),byteorder='little'))

		pop = toolbox.population(n=config['pop_size'])

		stats = tools.Statistics(lambda ind: ind.fitness.values)

		filter_feasible = lambda func, criteria: func(list(filter(is_feasible_fitness_criteria,criteria)))

		stats.register("min",lambda fit: filter_feasible(np.min, fit))
		stats.register("avg",lambda fit: filter_feasible(np.mean, fit))
		stats.register("max",lambda fit: filter_feasible(np.max, fit))

		logbook = tools.Logbook()
		logbook.header = ['gen','nevals'] + (stats.fields if stats else [])

		hof = tools.HallOfFame(config['hof_size'])

		run_filepath = f"{base_name}_{run_id}{ext}"

		print(f"[Proces {run_id}] "f"Rozpoczęcie ewolucji nr "f"{evolution_counter}")

		for gen in range(1,config['generations'] + 1):
			pop, _ = algorithms.eaSimple(
				pop,
				toolbox,
				cxpb=config['p_xov'],
				mutpb=config['p_mut'],
				ngen=1,
				stats=None,
				halloffame=hof,
				verbose=False
			)

			record = (stats.compile(pop) if stats else {})

			logbook.record(gen=gen,nevals=len(pop),**record)

			if len(hof) > 0:
				best_ind = hof[0]

				best_data = {
					"generation": gen,
					"fitness": best_ind.fitness.values,
					"genotype": str(best_ind[0])
				}

				with open(run_filepath,"a") as f:
					f.write(json.dumps(best_data)+ "\n")

		print(f"[Proces {run_id}] "f"Zakończono! "f"Zapisano do: {run_filepath}")
		evolution_counter += 1


def main():
	with open(os.path.join(project_path,"configs","generate_individual_data_config.yaml")) as f:
		config = yaml.safe_load(f)

	num_runs = 30

	tasks = [(i, config)for i in range(1, num_runs + 1)]

	num_workers = min(num_runs,multiprocessing.cpu_count())

	print(f"Uruchamianie {num_runs} eksperymentów "f"na {num_workers} procesach roboczych...")

	with multiprocessing.Pool(processes=num_workers) as pool:
		pool.map(run_single_experiment,tasks)

	print("Wszystkie równoległe przetwarzania zostały zakończone.")

if __name__ == "__main__":
	multiprocessing.freeze_support()
	main()
