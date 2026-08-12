from deap import tools
def run_cma_es_with_validation(toolbox, ngen, stats, halloffame, verbose=True, validity_threshold=0.2):
	logbook = tools.Logbook()
	logbook.header = ['gen', 'nevals', 'valid_recon_ratio', 'valid_frams_ratio'] + (stats.fields if stats else [])

	for gen in range(ngen):
		pop = toolbox.generate()
		results = toolbox.map(toolbox.evaluate, pop)

		valid_reconstruct_count = 0
		valid_framstick_count = 0

		for ind, (fit, is_valid_recon, is_valid_framsticks) in zip(pop, results):
			ind.fitness.values = fit
			if is_valid_recon:
				valid_reconstruct_count += 1
			if is_valid_framsticks:
				valid_framstick_count += 1

		valid_reconstruct_ratio = valid_reconstruct_count / len(pop)
		valid_framstick_ratio = valid_framstick_count / len(pop)

		toolbox.update(pop)

		if halloffame is not None:
			halloffame.update(pop)

		record = stats.compile(pop) if stats else {}
		logbook.record(gen=gen, nevals=len(pop), valid_recon_ratio=valid_reconstruct_ratio,valid_frams_ratio=valid_framstick_ratio, **record)

		if verbose:
			print(logbook.stream)

		# Obsługa spadku poprawności poniżej progu
		if valid_reconstruct_ratio < validity_threshold:
			print(f"\n[UWAGA] W generacji {gen} odsetek poprawnych genotypów spadł do {valid_reconstruct_ratio * 100:.1f}%.")

	return pop, logbook