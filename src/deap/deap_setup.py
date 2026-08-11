from deap import base, creator, tools, cma
from src.deap.native_operators import frams_evaluate, frams_crossover, frams_mutate, frams_getsimplest
from src.deap.selection import selTournament_only_feasible, selNSGA2_only_feasible

def prepare_native_toolbox(frams_lib, config: dict):
	opt_criteria = config['opt_criteria']

	if not hasattr(creator, "FitnessMax"):
		creator.create("FitnessMax", base.Fitness, weights=[1.0] * len(opt_criteria))
	if not hasattr(creator, "Individual"):
		creator.create("Individual", list, fitness=creator.FitnessMax)

	toolbox = base.Toolbox()
	toolbox.register("attr_simplest", frams_getsimplest, frams_lib, config.get('genformat', '1'),
					 config.get('initial_genotype'))
	toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_simplest, 1)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)

	toolbox.register("evaluate", frams_evaluate, frams_lib, opt_criteria, config)
	toolbox.register("mate", frams_crossover, frams_lib)
	toolbox.register("mutate", frams_mutate, frams_lib)

	if len(opt_criteria) <= 1:
		toolbox.register("select", selTournament_only_feasible, tournsize=config.get('tournament_size', 5))
	else:
		toolbox.register("select", selNSGA2_only_feasible)

	return toolbox

def prepare_cmaes_toolbox(frams_lib, config: dict):
	opt_criteria = config['opt_criteria']

	if not hasattr(creator, "FitnessMax"):
		creator.create("FitnessMax", base.Fitness, weights=[1.0] * len(opt_criteria))
	if not hasattr(creator, "Individual"):
		creator.create("Individual", list, fitness=creator.FitnessMax)

	strategy = cma.Strategy(centroid=[config['centroid']] * config['latent_dim'], sigma= config['sigma'], lambda_=config['pop_size'])

	toolbox = base.Toolbox()
	toolbox.register("generate",strategy.generate,creator.Individual)
	toolbox.register("update",strategy.update)
	return toolbox