import mip
from mip.constants import CutType
from utils.data_set import extract_data_set_info, get_data_json
from utils.matrix import fix_diagonal_weight
from loguru import logger
from typing import List, Optional, Tuple
from networkx import DiGraph, minimum_cut
from itertools import product
import numpy as np


class BSS:
    model: mip.Model
    num_vertices: int
    demands: np.ndarray
    vehicle_capacity: int
    distance_matrix: np.matrix
    num_vehicle: int
    x: List[List[mip.Var]] = []  # variables matrix

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
        self.model = mip.Model(name="BSS", sense=mip.MINIMIZE, solver_name=mip.CBC)

    def setup_variables(self) -> None:
        # Setting variables name

        def var_name(i: int, j: int) -> str:
            return f"x_{i}_{j}"

        self.x = [
            [
                self.model.add_var(var_type=mip.BINARY, name=var_name(i, j))
                for j in range(self.num_vertices)
            ]
            for i in range(self.num_vertices)
        ]

    def set_objective_function(self) -> None:
        self.model.objective = mip.minimize(
            mip.xsum(
                self.distance_matrix[i, j] * self.x[i][j]
                for i in range(self.num_vertices)
                for j in range(self.num_vertices)
            )
        )

    def add_constraints(self) -> None:
        # Constraint: \sum_{i \in V} x_{ij} = 1, \forall j \in V \setminus  \{0\}

        for i in range(self.num_vertices):
            self.model += (
                mip.xsum(self.x[i][j] for j in range(1, self.num_vertices)) == 1,
                "every node is viseted once on arc (i, j) except depot",
            )

        # Constraint: \sum_{i \in V} x_{ji} = 1, \forall j \in V \setminus  \{0\}

        for i in range(self.num_vertices):
            self.model += (
                mip.xsum(self.x[j][i] for j in range(1, self.num_vertices)) == 1,
                "every node is visited once on arc (j, i) except depot",
            )

        # Constraint: \sum_{j \in V} x_{0j} <= m (num_vehicle)

        self.model += (
            mip.xsum(self.x[0][j] for j in range(self.num_vertices))
            <= self.num_vehicle,
            "most m vehicles leave depot",
        )

        # Constraint: \sum_{j \in V \setminus \{0\}} x_{0j} = \sum_{j \in V \setminus \{0\}} x_{j0}

        self.model += (
            mip.xsum(self.x[0][j] for j in range(1, self.num_vertices))
            == mip.xsum(self.x[j][0] for j in range(1, self.num_vertices)),
            "all vehicles that are used return to the depot at the end of their route",
        )

        # Constraint:
        # \sum_{i \in S} \sum_{j \in S} x_{ij} \leq |S| - \max\{1, \left\lceil \dfrac{|\sum_{i \in S} q_{i}|}{Q} \right\rceil\}, S \subseteq V \setminus \{0\}, S \neq \emptyset

    def adding_exponencial_constraints(self) -> None:
        new_constraints = True
        while new_constraints:
            self.model.optimize(relax=True)

            logger.success("Status: ")
            print(self.model.status)

            logger.success("Objective value: ")
            print(self.model.objective_value)

            G = DiGraph()

            for i in range(self.num_vertices):
                for j in range(self.num_vertices):
                    G.add_edge(i, j, capacity=self.distance_matrix[i, j])

            new_constraints = False
            for n1, n2 in [
                (i, j)
                for (i, j) in product(
                    range(self.num_vertices), range(self.num_vertices)
                )
                if i != j
            ]:
                cut_value, (S, _) = minimum_cut(G, n1, n2)

                if cut_value <= 0.99:
                    self.model += (
                        mip.xsum(
                            self.x[i][j] for i in S for j in S if self.x[i][j] in S
                        )
                        - (len(S) - 1)
                        <= 0
                    )

                    new_constraints = True

            if not new_constraints:
                cp = self.model.generate_cuts(
                    [
                        mip.CutType.GOMORY,
                        CutType.MIR,
                        CutType.ZERO_HALF,
                        CutType.KNAPSACK_COVER,
                    ]
                )

                if cp.cuts:
                    self.model += cp
                    new_constraints = True

    def solve(self) -> Optional[List[Tuple[int, int]]]:
        pass
        # try:
        #     self.model.solve()
        #     solution = self.model.solution.get_values()
        #     selected_edges = [
        #         (i, j)
        #         for i in range(self.num_vertices)
        #         for j in range(self.num_vertices)
        #         if solution[self.model.variables.get_indices(self.x[i][j])] == 1
        #     ]

        #     return selected_edges
        # except CplexError as error:
        #     logger.exception(error)
        #     return None


def main() -> None:
    try:
        data = get_data_json("../../data/Bari/1Bari30.json")
        (num_vertices, demands, vehicle_capacity, distance_matrix) = (
            extract_data_set_info(ds=data)
        )
        fix_diagonal_weight(distance_matrix=distance_matrix)
        print(distance_matrix)
        problem = BSS(num_vertices, demands, vehicle_capacity, distance_matrix)
        problem.setup_model()
        problem.setup_variables()
        problem.set_objective_function()

        problem.add_constraints()
        problem.adding_exponencial_constraints()

        for i in range(0, 13):
            for j in range(0, 13):
                if problem.x[i][j].x >= 0.99:
                    print("{} -> {}".format(i, j))

        # s = problem.solve()
    except Exception as e:
        logger.exception(e)
