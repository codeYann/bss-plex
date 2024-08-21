from model.model import BSS
from loguru import logger
from typing import Set, Dict, Tuple
from mip import CutType
from networkx import minimum_cut, DiGraph
from itertools import product
from threading import Lock
import numpy as np
import mip

logger_lock = Lock()


def thread_safe_log(message: str, level="info"):
    with logger_lock:
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "success":
            logger.success(message)
        elif level == "exception":
            logger.exception(message)


class Flow(BSS):
    def __init__(
        self,
        V: Set[int],
        A: Dict[Tuple[int, int], int],
        m: int,
        Q: int,
        q: np.ndarray,
        c: np.matrix,
    ) -> None:
        super().__init__(V, A, m, Q, q, c)

        self.f = {
            (i, j): self.model.add_var(
                name="f_{}_{}".format(i, j), var_type=mip.CONTINUOUS, lb=0
            )
            for i, j in A
        }

        self._adding_aditional_constraints()

    def _adding_aditional_constraints(self) -> None:
        for j in self.V - {0}:
            self.model += (
                mip.xsum(self.f[j, i] for i in self.V)
                - mip.xsum(self.f[i, j] for i in self.V)
                == self.q[j]
            )

        # for i, j in self.A:
        #     max_flow = max(0, self.q[i], -self.q[j])
        #     min_flow = min(self.Q, self.Q + self.q[i], self.Q - self.q[j])

        #     self.model += (
        #         max_flow * self.x[i, j] <= self.f[i, j],
        #         f"flow_min_{i}_{j}",
        #     )

        #     self.model += (
        #         self.f[i, j] <= min_flow * self.x[i, j],
        #         f"flow_max_{i}_{j}",
        #     )

    def _cutting_plane(self):
        constraints = True

        while constraints:
            self.model.optimize(relax=True)

            print(
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

    def solve(self, log_path: str) -> None:
        logger.add(
            log_path, colorize=True, format="{time} {level} {message}", level="INFO"
        )

        status, objective_value = self._cutting_plane()

        if status == mip.OptimizationStatus.OPTIMAL:
            thread_safe_log("Success on finding optimal solution", "success")
            thread_safe_log(f"Objective value: {objective_value}", "info")

            for var in self.model.vars:
                thread_safe_log(f"{var.name} = {var.x}", "info")
        elif status == mip.OptimizationStatus.FEASIBLE:
            thread_safe_log("Feasible solution found", "info")
            thread_safe_log(f"Objective value: {objective_value}", "info")

            for var in self.model.vars:
                thread_safe_log(f"{var.name} = {var.x}", "info")
        else:
            thread_safe_log("Not found optimal solution", "warning")
