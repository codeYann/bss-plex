from typing import List, Dict, Tuple, Set
import numpy as np


def calculate_cumulative_demand(
    path: List[int], demands: Dict[int, int]
) -> Tuple[int | float, int | float]:
    """
    Calculate the minimum and maximum cumulative demand along a path.
    """
    cumulative = 0
    q_min, q_max = float("inf"), float("-inf")

    for i in range(1, len(path)):
        vertex = path[i]
        cumulative += demands[vertex]
        q_min = min(q_min, cumulative)
        q_max = max(q_max, cumulative)

    return q_min, q_max


def is_path_feasible(path: List[int], demands: Dict[int, int], capacity: int) -> bool:
    """
    Check if a path is feasible given the demands and capacity constraints.
    """
    if len(path) == 0:
        raise ValueError("A path must have, at least, 2 vertices")
    q_min, q_max = calculate_cumulative_demand(path, demands)
    return q_max - q_min <= capacity


def find_closest_unserved_customer(
    current: int, customers: List[int], distances: np.matrix
) -> int:
    """
    Find the closest unserved customer from the current position.
    """
    if not customers:
        return -1
    unserved_distances = distances[current, customers]
    closest_index = np.argmin(unserved_distances)
    return customers[closest_index]


def closest_neighbor(
    depot: int,
    customers: List[int],
    demands: Dict[int, int],
    distances: np.matrix,
    capacity: int,
) -> List[List[int]]:
    """
    Implement the closest neighbor heuristic for vehicle routing.
    Constructs routes by repeatedly adding the closest feasible customer.
    """
    routes: List[List[int]] = []
    visited: Set[int] = set()

    while customers:
        route = [depot]
        current = depot
        while True:
            next_customer = find_closest_unserved_customer(
                current, customers, distances
            )

            if next_customer == -1:
                break

            extended_route = route.copy()
            extended_route.append(next_customer)

            if not is_path_feasible(extended_route, demands, capacity):
                del extended_route
                break

            route.append(next_customer)
            visited.add(next_customer)
            customers.remove(next_customer)
            current = next_customer

        route.append(depot)
        routes.append(route)
    return routes
