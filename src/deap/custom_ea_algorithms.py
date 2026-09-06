from deap import tools
from utils.FramsticksPostProcessor import PostProcessFlag
def run_cma_es_with_validation(toolbox, ngen, stats, halloffame, verbose=True):
	logbook = tools.Logbook()
	logbook.header = ['gen', 'nevals', 'valid_recon_ratio','F_ZERO_LEN','F_SUBGROUPS','F_LONG_PARTS', 'valid_frams_ratio'] + (stats.fields if stats else [])
	created_individuals = []

	for gen in range(ngen):
		pop = toolbox.generate()
		results = toolbox.map(toolbox.evaluate, pop)

		valid_reconstruct_count = 0
		valid_framstick_count = 0
		invalid_zero_length_count = 0
		invalid_subgroups_count = 0
		invalid_long_parts_count = 0

		for ind, (fit, is_valid_recon, is_valid_framsticks, genotype, flags) in zip(pop, results):
			ind.fitness.values = fit
			ind.genotype = genotype
			if is_valid_recon:
				valid_reconstruct_count += 1
			if is_valid_framsticks:
				valid_framstick_count += 1
			if is_valid_recon and is_valid_framsticks:
				created_individuals.append(ind)
			if PostProcessFlag.INVALID_ZERO_LENGTH_JOINTS in flags:
				invalid_zero_length_count += 1
			if PostProcessFlag.INVALID_SUBGROUPS in flags:
				invalid_subgroups_count += 1
			if PostProcessFlag.INVALID_TO_LONG_PARTS in flags:
				invalid_long_parts_count += 1


		valid_reconstruct_ratio = round(valid_reconstruct_count / len(pop), 2)
		valid_framstick_ratio = round(valid_framstick_count / len(pop),2)
		invalid_zero_length_ratio = round(invalid_zero_length_count / len(pop),2)
		invalid_subgroups_ratio = round(invalid_subgroups_count / len(pop),2)
		invalid_long_parts_ratio = round(invalid_long_parts_count / len(pop),2)

		toolbox.update(pop)

		if halloffame is not None:
			halloffame.update(pop)
		record = stats.compile(pop) if stats else {}
		logbook.record(
			gen=gen,
			nevals=len(pop),
			valid_recon_ratio=valid_reconstruct_ratio,
			F_ZERO_LEN=invalid_zero_length_ratio,
			F_SUBGROUPS=invalid_subgroups_ratio,
			F_LONG_PARTS=invalid_long_parts_ratio,
			valid_frams_ratio=valid_framstick_ratio, **record)

		if verbose:
			print(logbook.stream)

		# # Obsługa spadku poprawności poniżej progu
		# if valid_reconstruct_ratio < validity_threshold:
		# 	reconstruction_ratio_above_threshold = False
		# 	break

	return pop, logbook, created_individuals