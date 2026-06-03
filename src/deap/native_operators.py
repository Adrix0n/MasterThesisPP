from src.deap.constraints import FITNESS_VALUE_INFEASIBLE_SOLUTION, genotype_within_constraint

def frams_evaluate(frams_lib, opt_criteria: list, config: dict, individual):
	FITNESS_CRITERIA_INFEASIBLE_SOLUTION = [FITNESS_VALUE_INFEASIBLE_SOLUTION] * len(opt_criteria)
	genotype = individual[0]
	data = frams_lib.evaluate([genotype])

	valid = True
	fitness = FITNESS_CRITERIA_INFEASIBLE_SOLUTION

	try:
		evaluation_data = data[0]["evaluations"][""]
		fitness = [evaluation_data[crit] for crit in opt_criteria]
	except (KeyError, TypeError, IndexError):
		valid = False

	if valid:
		evaluation_data['numgenocharacters'] = len(genotype)
		# Pobieranie limitów ze słownika konfiguracji
		valid &= genotype_within_constraint(genotype, evaluation_data, 'numparts', config.get('max_numparts'))
		valid &= genotype_within_constraint(genotype, evaluation_data, 'numjoints', config.get('max_numjoints'))
		valid &= genotype_within_constraint(genotype, evaluation_data, 'numneurons', config.get('max_numneurons'))
		valid &= genotype_within_constraint(genotype, evaluation_data, 'numconnections',
											config.get('max_numconnections'))
		valid &= genotype_within_constraint(genotype, evaluation_data, 'numgenocharacters',
											config.get('max_numgenochars'))

	if not valid:
		fitness = FITNESS_CRITERIA_INFEASIBLE_SOLUTION

	return tuple(fitness)


def frams_crossover(frams_lib, individual1, individual2):
	geno1, geno2 = individual1[0], individual2[0]
	individual1[0] = frams_lib.crossOver(geno1, geno2)
	individual2[0] = frams_lib.crossOver(geno1, geno2)
	return individual1, individual2


def frams_mutate(frams_lib, individual):
	individual[0] = frams_lib.mutate([individual[0]])[0]
	return individual,


def frams_getsimplest(frams_lib, genetic_format, initial_genotype):
	return initial_genotype if initial_genotype is not None else frams_lib.getSimplest(genetic_format)