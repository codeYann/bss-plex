import math
import mip
import networkx as nx
import numpy as np
from heuristics.closest_neighbor import closest_neighbor
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
        self,
        V: Set[int],
        A: Dict[Tuple[int, int], int],
        demands: List[int],
        vehicle_capacity: int,
    ) -> nx.DiGraph:
        x = self.x
        vertices = V
        arcs = {(i, j): x[i, j].x for (i, j) in A if x[i, j]}

        print(arcs)
        G = nx.DiGraph()

        for vertice in vertices:
            G.add_node(vertice)

        for (u, v), capacity in arcs.items():
            G.add_edge(u, v, capacity=capacity)

        u = len(vertices) + 1  # represents vertice n + 1
        v = len(vertices) + 2  # represents vertice n + 2

        G.add_nodes_from([u, v])

        for vertice, demand in enumerate(demands):
            if demand > 0:
                G.add_edge(u, vertice, capacity=demand / vehicle_capacity)
            else:
                G.add_edge(v, vertice, capacity=-demand / vehicle_capacity)

        return G

    def generate_constrs(self, model: mip.Model, depth: int = 0, npass: int = 0):
        vertices, arcs, demands, vehicle_capacity, cut_pool = (
            self.V,
            self.A,
            self.q.copy(),
            self.Q,
            mip.CutPool(),
        )

        # translate ownership model reference from x to y
        y = model.translate(self.x)
        G_prime = nx.DiGraph()

        # Creating arcs of G_prime graph
        arcs_prime = {(i, j): y[i][j].x for (i, j) in arcs if y[i][j] and i != j}

        # Creating G_prime graph
        for vertice in vertices:
            G_prime.add_node(vertice)

        for (u, v), capacity in arcs_prime.items():
            G_prime.add_edge(u, v, capacity=capacity)

        u = len(vertices) + 1  # represents vertice n + 1
        v = len(vertices) + 2  # represents vertice n + 2

        G_prime.add_nodes_from([u, v])

        for vertice, demand in enumerate(demands):
            if demand > 0:
                G_prime.add_edge(u, vertice, capacity=demand / vehicle_capacity)
            else:
                G_prime.add_edge(v, vertice, capacity=-demand / vehicle_capacity)

        # Adding q_{0} = Qtot
        demands[0] = sum(demands)

        flow_value, _ = nx.maximum_flow(G_prime, u, v)
        if flow_value < 1:
            _, (S, _) = nx.minimum_cut(G_prime, u, v)
            S = S - {0}
            if len(S) != 0:
                sum_demands = abs(sum(demands[i] for i in S))
                min_vehicles = math.ceil(sum_demands / vehicle_capacity)

                cut = mip.xsum(
                    self.x[i, j] for i in S for j in vertices - S if (i, j) in self.x
                ) <= len(S) - max(1, min_vehicles)

                print(cut)

                cut_pool.add(cut)

        for cut in cut_pool.cuts:
            model += cut


def generate_initial_solution(
    stations: Set[int],
    demands: List[int],
    distance_matrix: np.matrix,
    vehicle_capacity: int,
    x: Dict[Tuple[int, int], mip.Var],
) -> List[Tuple[mip.Var, int]]:
    depot = 0
    customers = list(stations - {depot})
    demand_dict = dict(enumerate(demands))
    routes = closest_neighbor(
        depot, customers, demand_dict, distance_matrix, vehicle_capacity
    )

    initial_solution: List[Tuple[mip.Var, int]] = []
    for route in routes:
        for i, j in zip(route, route[1:]):
            initial_solution.append((x[i, j], 1))

    return initial_solution


def print_solution(x: Dict[Tuple[int, int], mip.Var]) -> None:
    print("Solution:")
    for (i, j), var in x.items():
        if var.x != 0:  # Assuming binary variables
            print(f"Route from {i} -> {j}")


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
        model.cuts_generator = ConstraintsGeneratorCallback(V, A, Q, q, x)
        model.optimize()

        # if model.num_solutions:
        #     model.check_optimization_results()

        #     optimal_route = [
        #         (origin, destination)
        #         for origin in V
        #         for destination in V
        #         if x[origin][destination].x >= 0.5
        #     ]

        #     for origin, destination in optimal_route:
        #         variable_name = f"x_{origin}_{destination}"
        #         variable_value = x[origin][destination].x
        #         print(f"{model.var_by_name(variable_name)} = {variable_value}")
        # else:
        #     print("Fuck")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
