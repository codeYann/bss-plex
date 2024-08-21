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


class Mtz(BSS):
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

        # Adding additional continuos variables t j for all j in V

        self.t = [
            self.model.add_var(
                "t_{}".format(j),
                var_type=mip.CONTINUOUS,
                lb=max(0, self.q[j]),
                ub=min(self.Q, self.Q + self.q[j]),
            )
            for j in self.V
        ]

        self._adding_aditional_constraints()

    def _adding_aditional_constraints(self) -> None:
        # for j in self.V:
        #     max_t = max(0, self.q[j])
        #     min_t = min(self.Q, self.Q + self.q[j])
        #     self.model += max_t <= self.t[j]
        #     self.model += self.t[j] <= min_t

        for i in self.V:
            for j in self.V:
                if j != 0:
                    M = min(self.Q, self.Q + self.q[j])
                    self.model += self.t[j] >= self.t[i] + self.q[j] - M * (
                        1 - self.x[i][j]
                    )

        for i in self.V:
            for j in self.V:
                if i != 0:
                    M = min(self.Q, self.Q - self.q[j])
                    self.model += self.t[i] >= self.t[j] - self.q[j] - M * (
                        1 - self.x[i][j]
                    )

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
                G.add_edge(i, j, capacity=self.x[i][j].x)

            constraints = False

            for u, v in [
                (i, j) for (i, j) in product(list(self.V), list(self.V)) if i != j
            ]:
                cut_value, (S, _) = minimum_cut(G, u, v)
                S = S - {0}

                if cut_value <= 0.99:
                    self.model += (
                        mip.xsum(self.x[i][j] for i, j in self.A if (i in S and j in S))
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

        route = [(i, j) for i in self.V for j in self.V if self.x[i][j].x >= 0.99]
        t = [self.t[j].x for j in self.V if self.t[j].x != 0]

        if status == mip.OptimizationStatus.OPTIMAL:
            thread_safe_log("Success on finding optimal solution", "success")
            thread_safe_log(f"Objective value: {objective_value}", "info")

            for j in range(len(t)):
                thread_safe_log(f"t_{j}: {t[j]}", "info")

            for i, j in route:
                thread_safe_log(
                    f"{self.model.var_by_name(f"x_{i}_{j}")} = {self.x[i][j].x}", "info"
                )

        elif status == mip.OptimizationStatus.FEASIBLE:
            thread_safe_log("Feasible solution found", "info")
            thread_safe_log(f"Objective value: {objective_value}", "info")

            for i, j in route:
                thread_safe_log(
                    f"{self.model.var_by_name(f"x_{i}_{j}")} = {self.x[i][j].x}", "info"
                )
        else:
            thread_safe_log("Not found optimal solution", "warning")
