import mip
from mip.constants import CutType
import numpy as np
from typing import Set, Dict, Tuple
from loguru import logger
from networkx import DiGraph, minimum_cut
from itertools import product


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
        A: Dict[Tuple[int, int], int],
        m: int,
        Q: int,
        q: np.ndarray,
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

        # Initialize the MIP model
        self.model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)

        # Set decisions variables
        self.x = {
            a: self.model.add_var(
                name="x_{}_{}".format(a[0], a[1]), var_type=mip.BINARY
            )
            for a in A
        }

        self._set_objective()
        self._add_constraints()

    def _set_objective(self) -> None:
        """
        Sets the objective function to minimize the total cost.
        """
        self.model.objective = mip.xsum(
            self.c[i, j] * self.x[i, j] for (i, j) in self.A
        )

    def _add_constraints(self) -> None:
        # Constraint: Every node besides depot is visited once.
        for i in self.V:
            self.model += (
                mip.xsum(self.x[i, j] for j in self.V - {0} if (i, j) in self.A) == 1,
                "every node is viseted once on arc (i, j) except depot",
            )

        for i in self.V:
            self.model += (
                mip.xsum(self.x[j, i] for j in self.V - {0} if (i, j) in self.A) == 1,
                "every node is viseted once on arc (j, i) except depot",
            )

        # Constraint: At most m vehicle leave the depot
        self.model += (
            mip.xsum(self.x[0, j] for j in self.V if (0, j) in self.A) <= self.m,
            "at most m vehicle leave the depot",
        )

        # Constraint: All vehicles used must return to the depot
        self.model += (
            mip.xsum(self.x[0, j] for j in self.V - {0} if (0, j) in self.A)
            == mip.xsum(self.x[j, 0] for j in self.V - {0} if (0, j) in self.A),
            "all vehicles used must return to the depot",
        )

    def _cutting_plane(self):
        logger.info("Processing cutting plane method")

        constraints = True

        while constraints:
            self.model.optimize(relax=True)

            logger.info(
                "Status: {}, Objective value: {}".format(
                    self.model.status, self.model.objective_value
                )
            )

            G = DiGraph()

            for a in self.A:
                G.add_edge(a[0], a[1], capacity=self.x[a].x)

            constraints = False

            for u, v in [(i, j) for (i, j) in product(self.V, self.V) if i != j]:
                cut_value, (S, NS) = minimum_cut(G, u, v)

                if cut_value <= 0.99:
                    self.model += mip.xsum(
                        self.x[a] for a in self.A if (a[0] in S and a[1] in S)
                    ) <= len(S) - max(
                        1, np.ceil((abs(sum(self.q[i] for i in S))) / self.Q)
                    )
                    constraints = True

            if not constraints and self.model.solver_name.lower() == "cbc":
                cp = self.model.generate_cuts(
                    [CutType.GOMORY, CutType.MIR, CutType.KNAPSACK_COVER]
                )
                if cp.cuts:
                    self.model += cp
                    constraints = True

        return self.model.status, self.model.objective_values

    def solve(self) -> None:
        """
        Solves the BSS problem and prints the solution.
        """
        status, objective_values = self._cutting_plane()

        print(status, objective_values)

        logger.success("Optimal solution found:")

        for i, j in self.A:
            if self.x[i, j].x >= 1:
                print("Route {} -> {}".format(i, j))
