FITNESS_VALUE_INFEASIBLE_SOLUTION = -999999.0

def is_feasible_fitness_value(fitness_value: float) -> bool:
    assert isinstance(fitness_value, float), f"Fitness must be float, got {type(fitness_value)}"
    return fitness_value != FITNESS_VALUE_INFEASIBLE_SOLUTION

def is_feasible_fitness_criteria(fitness_criteria: tuple) -> bool:
    return all(is_feasible_fitness_value(val) for val in fitness_criteria)

def genotype_within_constraint(genotype: str, dict_criteria_values: dict, criterion_name: str, constraint_value: float) -> bool:
    if constraint_value is not None:
        actual_value = dict_criteria_values.get(criterion_name)
        if actual_value is not None and actual_value > constraint_value:
            return False
    return True