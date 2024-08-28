import mip
import numpy as np
from typing import Set, Dict, Tuple, Any, List


class BSS:
    """
    Bike Sharing System (BSS) as a Vehicle Routing Problem (VRP).

    Inputs:
    - V: Set of vertices (stations)
    - A: Set of arcs represented as a dictionary with (i, j) as keys and cost as values
    - m: Number of vehicles
    - Q: Vehicle capacity
    - q: Demand at each vertex (list)
    - c: Cost matrix representing the cost of traveling between stations
    """

    def __init__(
        self,
        V: Set[int],
        A: Dict[Tuple[int, int], Any],
        m: int,
        Q: int,
        q: np.ndarray | List[int],
        c: np.matrix,
    ) -> None:
        self.V = V
        self.A = A
        self.n = len(V)
        self.m = m
        self.Q = Q
        self.q = q
        self.c = c

        # Validate inputs
        assert len(q) == self.n, "Demand list size must match the number of vertices"

        # Initialize MIP model
        self.model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)

        # Set decisions variables
        self.x = [
            [
                self.model.add_var(name="x_{}_{}".format(i, j), var_type=mip.BINARY)
                for i in self.V
            ]
            for j in self.V
        ]

        self.set_objective_function()
        self.base_constraints()

    def set_objective_function(self) -> None:
        """
        Sets the objective function to minimize the total cost.
        """
        self.model.objective = mip.minimize(
            mip.xsum(int(self.c[i, j]) * self.x[i][j] for i in self.V for j in self.V)
        )

    def base_constraints(self) -> None:
        # Constraint: Every node besides depot is visited once.

        self.model += mip.xsum(self.x[i][i] for i in self.V) == 0

        for i in self.V:
            self.model += (
                mip.xsum(self.x[i][j] for j in self.V if j != 0) == 1,
                "every node is viseted once on arc (i, j) except depot",
            )
            self.model += (
                mip.xsum(self.x[j][i] for j in self.V if j != 0) == 1,
                "every node is viseted once on arc (j, i) except depot",
            )

        # Constraint: At most m vehicle leave the depot
        self.model += (
            mip.xsum(self.x[0][j] for j in self.V) <= self.m,
            "at most m vehicle leave the depot",
        )

        # Constraint: All vehicles used must return to the depot
        self.model += (
            mip.xsum(self.x[0][j] for j in self.V if j != 0)
            - mip.xsum(self.x[j][0] for j in self.V if j != 0)
            == 0,
            "all vehicles used must return to the depot",
        )
