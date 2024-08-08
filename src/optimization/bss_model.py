import cplex
from cplex.exceptions import CplexError
from utils import utils
from loguru import logger
from typing import List, Optional, Tuple
import numpy as np
import itertools


class BSS:
    model: cplex.Cplex = None
    num_vertices: int
    demands: np.ndarray
    vehicle_capacity: int
    distance_matrix: np.matrix
    num_vehicle: int
    x: List[List[str]] = []

    def __init__(
        self,
        num_vertices: int,
        demands: np.ndarray,
        vehicle_capacity: int,
        distance_matrix: np.matrix,
        num_vehicle=5,
    ):
        self.num_vertices = num_vertices
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.distance_matrix = distance_matrix
        self.num_vehicle = num_vehicle

    def setup_model(self) -> None:
        self.model = cplex.Cplex()
        self.model.set_problem_name("BSS")
        self.model.set_problem_type(cplex.Cplex.problem_type.MILP)
        self.model.set_results_stream(None)

    def setup_variables(self) -> None:
        for i in range(self.num_vertices):
            self.x.append([f"x_{i}_{j}" for j in range(self.num_vertices)])

        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                if i != j:
                    self.model.variables.add(
                        obj=[self.distance_matrix[i, j]],
                        lb=[0],
                        ub=[1],
                        types=[self.model.variables.type.binary],
                        names=[self.x[i][j]],
                    )
                else:
                    self.model.variables.add(
                        lb=[0],
                        ub=[0],
                        types=[self.model.variables.type.binary],
                        names=[self.x[i][j]],
                    )

    def add_constraints(self) -> None:
        # Constraint: \sum_{i \in V} x_{ij} = 1, \forall j \in V \setminus  \{0\}

        for i in range(self.num_vertices):
            self.model.linear_constraints.add(
                lin_expr=[
                    cplex.SparsePair(
                        ind=[self.x[i][j] for j in range(1, self.num_vertices)],
                        val=[1] * (self.num_vertices - 1),
                    )
                ],
                senses=["E"],
                rhs=[1],
            )

        # Constraint: \sum_{i \in V} x_{ji} = 1, \forall j \in V \setminus  \{0\}

        for i in range(self.num_vertices):
            self.model.linear_constraints.add(
                lin_expr=[
                    cplex.SparsePair(
                        ind=[self.x[j][i] for j in range(1, self.num_vertices)],
                        val=[1] * (self.num_vertices - 1),
                    )
                ],
                senses=["E"],
                rhs=[1],
            )

        # Constraint: \sum_{j \in V} x_{0j} <= m (num_vehicle)

        self.model.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[self.x[0][j] for j in range(1, self.num_vertices)],
                    val=[1] * (self.num_vertices - 1),
                )
            ],
            senses=["L"],
            rhs=[self.num_vehicle],
        )

        # Constraint: \sum_{j \in V \setminus \{0\}} x_{0j} = \sum_{j \in V \setminus \{0\}} x_{j0}

        self.model.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[self.x[0][j] for j in range(1, self.num_vertices)]
                    + [self.x[j][0] for j in range(1, self.num_vertices)],
                    val=([1] * (self.num_vertices - 1))
                    + ([-1] * (self.num_vertices - 1)),
                )
            ],
            senses=["E"],
            rhs=[0],
        )

        # Constraint:
        # \sum_{i \in S} \sum_{j \in S} x_{ij} \leq |S| - \max\{1, \left\lceil \dfrac{|\sum_{i \in S} q_{i}|}{Q} \right\rceil\}, S \subseteq V \setminus \{0\}, S \neq \emptyset

        for size in range(2, self.num_vertices):
            for S in itertools.combinations(range(1, self.num_vertices), size):
                demand_sum = np.sum(self.demands[i] for i in S)
                max_vehicles = max(1, int(np.ceil(demand_sum / self.vehicle_capacity)))
                S_list = list(S)

                self.model.linear_constraints.add(
                    lin_expr=[
                        cplex.SparsePair(
                            ind=[
                                self.x[i][j] for i in S_list for j in S_list if i != j
                            ],
                            val=[1]
                            * len([1 for i in S_list for j in S_list if i != j]),
                        )
                    ],
                    senses=["L"],
                    rhs=[len(S_list) - max_vehicles],
                )

    def solve(self) -> Optional[List[Tuple[int, int]]]:
        try:
            self.model.solve()
            solution = self.model.solution.get_values()
            selected_edges = [
                (i, j)
                for i in range(self.num_vertices)
                for j in range(self.num_vertices)
                if solution[self.model.variables.get_indices(self.x[i][j])] == 1
            ]

            return selected_edges
        except CplexError as error:
            logger.exception(error)
            return None


def main() -> None:
    try:
        data = utils.get_data_json("../../data/Bari/1Bari30.json")
        (num_vertices, demands, vehicle_capacity, distance_matrix) = (
            utils.extract_data_set_info(ds=data)
        )
        problem = BSS(num_vertices, demands, vehicle_capacity, distance_matrix)
        problem.setup_model()
        problem.setup_variables()
        problem.add_constraints()

        s = problem.solve()
    except Exception as e:
        logger.exception(e)
