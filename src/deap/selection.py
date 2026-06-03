from deap import tools
from src.deap.constraints import is_feasible_fitness_criteria

def select_feasible(individuals):
    # Filtruje tylko poprawne rozwiązania z populacji.
    feasible_individuals = [ind for ind in individuals if is_feasible_fitness_criteria(ind.fitness.values)]
    return feasible_individuals

def selTournament_only_feasible(individuals, k, tournsize):
    return tools.selTournament(select_feasible(individuals), k, tournsize=tournsize)

def selNSGA2_only_feasible(individuals, k):
    return tools.selNSGA2(select_feasible(individuals), k)