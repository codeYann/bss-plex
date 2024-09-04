import math
import mip
import networkx as nx
import numpy as np
from model.heuristics import closest_neighbor
from utils.data_set import extract_data_set_info, get_data_json
from utils.matrix import fix_diagonal_weight
from typing import Dict, List, Set, Tuple


def create_distance_dict(distance_matrix: np.matrix) -> Dict[Tuple[int, int], int]:
    return {
        (i, j): int(distance_matrix[i, j])
        for i in range(len(distance_matrix))
        for j in range(len(distance_matrix))
    }


def import_data(
    file_path: str,
) -> Tuple[Set[int], Dict[Tuple[int, int], int], int, int, List[int], np.matrix]:
    data_set = get_data_json(file_path)
    vertices, demands, vehicle_capacity, distance_matrix = extract_data_set_info(
        data_set
    )

    fix_diagonal_weight(distance_matrix)

    return (
        set(range(vertices)),
        create_distance_dict(distance_matrix),
        1,  # Number of vehicles
        vehicle_capacity,
        demands,
        distance_matrix,
    )


class ConstraintsGeneratorCallback(mip.ConstrsGenerator):
    def __init__(
        self,
        V: Set[int],
        A: Dict[Tuple[int, int], int],
        Q: int,
        q: List[int],
        x: Dict[Tuple[int, int], mip.Var],
    ) -> None:
        self.V = V
        self.A = A
        self.Q = Q
        self.q = q
        self.x = x

    def _create_supporting_graph(
        self, V: Set[int], A: Dict[Tuple[int, int], int]
    ) -> nx.DiGraph:
        x = self.x
        vertices = V
        arcs = {(i, j): x[i, j].x for (i, j) in A if x[i, j].x != 0}

        G = nx.DiGraph()

        for vertice in vertices:
            G.add_node(vertice)

        for (u, v), capacity in arcs.items():
            G.add_edge(u, v, capacity=capacity)

        G.add_node()

    def generate_constrs(self, model: mip.Model, depth: int = 0, npass: int = 0):
        pass


def generate_initial_solution(
    stations: Set[int],
    demands: List[int],
    distance_matrix: np.matrix,
    vehicle_capacity: int,
    x: Dict[Tuple[int, int], mip.Var],
) -> List[Tuple[mip.Var, float]]:
    depot = 0
    customers = list(stations - {depot})
    demand_dict = dict(enumerate(demands))
    routes = closest_neighbor(
        depot, customers, demand_dict, distance_matrix, vehicle_capacity
    )

    initial_solution = []
    for route in routes:
        for i, j in zip(route, route[1:]):
            initial_solution.append((x[i, j], 1))

    return initial_solution


def main() -> None:
    try:
        V, A, m, Q, q, c = import_data("../../data/ReggioEmilia/4ReggioEmilia30.json")

        model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)

        # Decision variable
        x: Dict[Tuple[int, int], mip.Var] = {
            (i, j): model.add_var(name=f"x_{i}_{j}", var_type=mip.BINARY)
            for (i, j) in A
        }

        # Setting objective function
        model.objective = mip.minimize(
            mip.xsum(cost * x[i, j] for (i, j), cost in A.items())
        )

        # Constraints
        for j in V - {0}:
            model += mip.xsum(x[i, j] for i in V) == 1  # Inflow
            model += mip.xsum(x[j, i] for i in V) == 1  # Outflow
        model += mip.xsum(x[0, j] for j in V - {0}) <= m  # Vehicle limit
        model += (
            mip.xsum(x[0, j] for j in V - {0}) - mip.xsum(x[j, 0] for j in V - {0}) == 0
        )

        # Generate an initial solution using closest neighbor heuristic
        initial_solution = generate_initial_solution(V, q, c, Q, x)

        model.start = initial_solution
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
