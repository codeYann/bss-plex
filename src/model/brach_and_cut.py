import numpy as np
import mip
from mip import ConstrsGenerator
from typing import Set, Tuple
from itertools import chain, combinations


class SubTourCutElimination(ConstrsGenerator):
    def __init__(
        self, V: Set[int], Q: int, q: np.ndarray, A: Set[Tuple[int, int]]
    ) -> None:
        self.V = V
        self.Q = Q
        self.A = A
        self.q = q

    def _generate_subsets(self, V: Set[int]):
        return chain.from_iterable(combinations(V, r) for r in range(1, len(V)))

    def generate_constrs(self, model: "mip.Model", depth: int = 0, npass: int = 0):
        for S in self._generate_subsets(self.V - {0}):
            total_demand = sum(self.q[i] for i in S)
            capacity_bound = max(1, int(np.ceil(np.abs(total_demand) / self.Q)))

            lhs = mip.xsum(model.vars[i, j] for i in S for j in S if (i, j) in self.A)
            rhs = len(S) - capacity_bound

            if lhs > rhs:
                model += lhs <= rhs
