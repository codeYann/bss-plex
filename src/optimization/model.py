import cplex
from cplex.exceptions import CplexError
from typing import List, Tuple
from loguru import logger


class Model:
    distance: List[List[int]]
    num_cities: int
    problem_instance: cplex.Cplex

    def __init__(self, distance: List[List[int]], num_cities: int) -> None:
        if len(distance) == 0:
            logger.error("Distance matrix can't be empty or null")

        self.distance = distance
        self.num_cities = num_cities
        self.problem_instance = None

    def set_problem_instance(self, instance: cplex.Cplex) -> None:
        self.problem_instance = instance

    def __create_tsp_model(self) -> cplex.Cplex:
        model = cplex.Cplex()

        model.set_problem_name("Classical TSP")
        model.set_problem_type(cplex.Cplex.problem_type.MILP)

        edges: List[Tuple[int, int]] = []

        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                edges.append((i, j))

        num_edges = len(edges)

        variable_names = [f"x_{i}_{j}" for i, j in edges]

        variable_types = ["B"] * num_edges

        variable_objectives = [self.distance[i][j] for i, j in edges]

        model.variables.add(
            obj=variable_objectives, names=variable_names, types=variable_types
        )

        # Contraints

        for i in range(self.num_cities):
            enter_contraints = [
                f"x_{min(i, j)}_{max(i, j)}" for j in range(self.num_cities) if i != j
            ]

            model.linear_constraints.add(
                lin_expr=[[enter_contraints, [1] * len(enter_contraints)]],
                senses=["E"],
                rhs=[1],
            )

            exit_constraint = [
                f"x_{min(i, j)}_{max(i, j)}" for j in range(self.num_cities) if i != j
            ]

            model.linear_constraints.add(
                lin_expr=[[exit_constraint, [1] * len(exit_constraint)]],
                senses=["E"],
                rhs=[1],
            )

        return model

    def solve(self) -> None:
        try:
            model = self.__create_tsp_model()
            model.solve()
            self.set_problem_instance(model)

            solution = model.solution

            logger.success("Objective value: ")
            logger.success(solution.get_objective_value())
            logger.info("Edges in solution:")

            for name, value in zip(model.variables.get_names(), solution.get_values()):
                if value >= 0.5:
                    logger.success(name)
        except CplexError as e:
            logger.exception(e)
