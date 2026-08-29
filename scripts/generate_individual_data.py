import yaml
import sys
import os
import json
import numpy as np
import random
import multiprocessing
from deap import tools, algorithms

# Dodanie ścieżki, aby działało środowisko
framspy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'external', 'framspy'))

if framspy_path not in sys.path:
	sys.path.insert(0, framspy_path)
from FramsticksLib import FramsticksLib
from src.deap.deap_setup import prepare_native_toolbox
from src.deap.constraints import is_feasible_fitness_criteria


def run_single_experiment(args):
	"""Funkcja robocza wywoływana przez każdy proces."""
	run_id, config = args

	# 1. Resetowanie ziarna losowości dla każdego procesu,
	# aby uniknąć identycznych wyników w każdym pliku
	random.seed(run_id + int.from_bytes(os.urandom(4), byteorder='little'))
	np.random.seed(run_id + int.from_bytes(os.urandom(4), byteorder='little'))

	# 2. Inicjalizacja środowiska Framsticks (Musi być wewnątrz procesu!)
	frams_lib = FramsticksLib(config['frams_path'], config['frams_lib'], config['sim_file'])
	toolbox = prepare_native_toolbox(frams_lib, config)
	pop = toolbox.population(n=config['pop_size'])

	# 3. Ustawienie statystyk
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	filter_feasible = lambda func, criteria: func(list(filter(is_feasible_fitness_criteria, criteria)))
	stats.register("min", lambda fit: filter_feasible(np.min, fit))
	stats.register("avg", lambda fit: filter_feasible(np.mean, fit))
	stats.register("max", lambda fit: filter_feasible(np.max, fit))

	logbook = tools.Logbook()
	logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

	hof = tools.HallOfFame(config['hof_size'])

	# 4. Generowanie unikalnej nazwy pliku z dopisanym _num na końcu
	orig_filepath = config["result_filepath"]
	base_name, ext = os.path.splitext(orig_filepath)
	run_filepath = f"{base_name}_{run_id}{ext}"
	#
	# # Upewnienie się, że plik jest pusty na start (jeśli proces odpala się ponownie)
	# with open(run_filepath, "w") as f:
	# 	pass

	# 5. Pętla ewolucyjna
	print(f"[Proces {run_id}] Rozpoczęcie ewolucji...")
	for gen in range(1, config['generations'] + 1):
		pop, _ = algorithms.eaSimple(pop, toolbox,
									 cxpb=config['p_xov'],
									 mutpb=config['p_mut'],
									 ngen=1,
									 stats=None,
									 halloffame=hof,
									 verbose=True)

		record = stats.compile(pop) if stats else {}
		logbook.record(gen=gen, nevals=len(pop), **record)

		# Możesz zakomentować print, aby 20 procesów nie zaśmiecało konsoli jednocześnie
		# print(f"[Proces {run_id}] Gen {gen}:", logbook.stream)

		# Zapis najlepszego osobnika (z Hall of Fame) do pliku w każdej epoce
		if len(hof) > 0:
			best_ind = hof[0]  # Indeks 0 to all-time best
			best_data = {
				"generation": gen,
				"fitness": best_ind.fitness.values,
				"genotype": str(best_ind[0])
			}

			with open(run_filepath, "a") as f:
				f.write(json.dumps(best_data) + "\n")

	print(f"[Proces {run_id}] Zakończono! Zapisano do: {run_filepath}")
	return run_id


def main():
	# Wczytanie konfiguracji tylko raz w głównym procesie
	with open("../configs/generate_individual_data_config.yaml") as f:
		config = yaml.safe_load(f)

	# Liczba przetwarzań (eksperymentów)
	num_runs = os.cpu_count() - 1


	# Przygotowanie argumentów dla każdego procesu (ID procesu od 1 do 20, konfiguracja)
	tasks = [(i, config) for i in range(1, num_runs + 1)]

	# Określenie liczby rdzeni do wykorzystania.
	# Bezpiecznie jest ograniczyć to do ilości dostępnych wątków CPU.
	num_workers = min(num_runs, multiprocessing.cpu_count())
	print(f"Uruchamianie {num_runs} eksperymentów na {num_workers} procesach roboczych...")

	# Uruchomienie puli procesów
	with multiprocessing.Pool(processes=num_workers) as pool:
		# map() zablokuje główny wątek do czasu zakończenia wszystkich 20 procesów
		pool.map(run_single_experiment, tasks)

	print("Wszystkie równoległe przetwarzania zostały zakończone.")


if __name__ == "__main__":
	# Konieczne w Windows do prawidłowego działania multiprocessing
	multiprocessing.freeze_support()
	main()