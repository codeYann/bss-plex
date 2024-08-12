import numpy as np


def fix_diagonal_weight(distance_matrix: np.matrix) -> None:
    np.fill_diagonal(a=distance_matrix, val=0)
