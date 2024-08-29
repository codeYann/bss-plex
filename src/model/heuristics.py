from typing import List, Dict, Tuple

def calculate_cumulative_demand(path: List[int], demands: Dict[int, int]) -> Tuple[int, int]:
    cumulative = 0
    q_min, q_max = float('inf'), float('-inf')

    for i in range(1, len(path)):
        vertex = path[i]
        cumulative += demands[vertex]
        q_min = min(q_min, cumulative)
        q_max = max(q_max, cumulative)
    return q_min, q_max

def is_path_feasible(path: List[int], demands: Dict[int, int], capacity: int) -> bool:
    if len(path) == 0:
        raise ValueError("A path must have at least 2 vertices")
    q_min, q_max = calculate_cumulative_demand(path, demands)
    return q_max - q_min <= capacity