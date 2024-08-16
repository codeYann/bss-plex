import mip
import numpy as np
from mip.constants import CutType
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

        # Initialize MIP model
        self.model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)

        # Set decisions variables
        self.x = {
            (i, j): self.model.add_var(
                name="x_{}_{}".format(i, j), var_type=mip.BINARY, lb=0, ub=1
            )
            for (i, j) in A
        }

        self._set_objective()
        self._add_constraints()

    def _set_objective(self) -> None:
        """
        Sets the objective function to minimize the total cost.
        """
        self.model.objective = mip.xsum(c * self.x[a] for a, c in self.A.items())

    def _add_constraints(self) -> None:
        # Constraint: Every node besides depot is visited once.
        for j in self.V - {0}:
            self.model += (
                mip.xsum(self.x[i, j] for i in self.V if (i, j) in self.A) == 1,
                "every node is viseted once on arc (i, j) except depot",
            )

            self.model += (
                mip.xsum(self.x[j, i] for i in self.V if (i, j) in self.A) == 1,
                "every node is viseted once on arc (j, i) except depot",
            )

        # Constraint: At most m vehicle leave the depot
        self.model += (
            mip.xsum(self.x[0, j] for j in self.V if (0, j) in self.A) <= self.m,
            "at most m vehicle leave the depot",
        )

        # Constraint: All vehicles used must return to the depot
        self.model += (
            mip.xsum(self.x[0, j] for j in (self.V - {0}) if (0, j) in self.A)
            == mip.xsum(self.x[j, 0] for j in (self.V - {0}) if (0, j) in self.A),
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

            for i, j in self.A:
                G.add_edge(i, j, capacity=self.x[i, j].x)

            constraints = False

            for u, v in [
                (i, j) for (i, j) in product(list(self.V), list(self.V)) if i != j
            ]:
                cut_value, (S, _) = minimum_cut(G, u, v)
                S = S - {0}

                if cut_value <= 0.99:
                    self.model += (
                        mip.xsum(self.x[i, j] for i, j in self.A if (i in S and j in S))
                        <= len(S) - 1
                    )

                    constraints = True

            if not constraints and self.model.solver_name.lower() == "cbc":
                cp = self.model.generate_cuts(
                    [
                        CutType.GOMORY,
                        CutType.MIR,
                        CutType.ZERO_HALF,
                        CutType.KNAPSACK_COVER,
                    ]
                )

                if cp.cuts:
                    self.model += cp
                    constraints = True

        return self.model.status, self.model.objective_value

    def solve(self) -> None:
        """
        Solves the BSS problem and prints the solution.
        """
        status, objective_value = self._cutting_plane()

        if status == mip.OptimizationStatus.OPTIMAL:
            logger.success("Success on finding optimal solution")
            logger.info("Objetive value: {}".format(objective_value))
        else:
            logger.info("Not found optimal solution")
            logger.info("Objetive value: {}".format(objective_value))
