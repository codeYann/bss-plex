import cplex
from cplex.exceptions import CplexError
from docplex.mp import Model
from utils import utils
from loguru import logger
from typing import List
import numpy as np


class BssModel:
    model: cplex.Cplex = None
    num_vertices: int
    demands: np.ndarray
    vehicle_capacity: int
    distance_matrix: np.matrix
    x: List[List[str]] = []

    def __init__(
        self,
        num_vertices: int,
        demands: np.ndarray,
        vehicle_capacity: int,
        distance_matrix: np.matrix,
    ):
        self.num_vertices = num_vertices
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.distance_matrix = distance_matrix

    def setup_model(self) -> None:
        self.model = cplex.Cplex()
        self.model.set_problem_name("BSS")
        self.model.set_problem_type(cplex.Cplex.problem_type.MILP)
        self.model.set_results_stream(None)

    def setup_variables(self) -> None:
        for i in range(self.num_vertices):
            self.x.append([f"x_{i}_{j}" for j in range(self.num_vertices)])
            for j in range(self.num_vertices):
                if i != j:
                    pass


def main() -> None:
    try:
        data = utils.get_data_json("../../data/Bari/1Bari30.json")
        (num_vertices, demands, vehicle_capacity, distance_matrix) = (
            utils.extract_data_set_info(ds=data)
        )
        problem = BssModel(num_vertices, demands, vehicle_capacity, distance_matrix)
        problem.setup_model()
    except Exception as e:
        logger.exception(e)
