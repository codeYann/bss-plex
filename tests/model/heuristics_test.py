import pytest
from src.model.heuristics import (
    calculate_cumulative_demand,
    is_path_feasible,
)


# Test calculate_cumulative_demand
def test_calculate_cumulative_demand():
    path = [0, 4, 3, 5, 2]
    demands = {0: 0, 3: -5, 5: 4, 2: 2, 4: -3}
    (q_min, q_max) = calculate_cumulative_demand(path, demands)
    assert q_min == -8
    assert q_max == -2


# Test is_path_feasible
def test_is_path_fesible():
    capacity = 10
    path = [0, 4, 3, 5, 2]
    demands = {0: 0, 3: -5, 5: 4, 2: 2, 4: -3}
    (q_min, q_max) = calculate_cumulative_demand(path, demands)
    (q_min, q_max) = calculate_cumulative_demand(path, demands)

    is_feasible = is_path_feasible(path, demands, capacity)

    assert q_max - q_min == 6
    assert is_feasible == True


# Test is_path_feasbile with empty path
def test_is_path_feasible_empty_path():
    with pytest.raises(ValueError):
        is_path_feasible([], {}, 10)

