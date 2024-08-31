import json
import numpy as np
from loguru import logger
from typing import Any, Dict, Tuple, List
import os


def get_data_json(path: str) -> Dict[str, Any]:
    """
    Reads a JSON file and returns the parsed data.

    Args:
        path (str): The file path to the JSON file.

    Returns:
        The parsed JSON data as a dictionary, or None if an error occurs.
    """
    base_path = os.path.dirname(__file__)
    abs_file_path = os.path.join(base_path, path)
    try:
        with open(abs_file_path, "r") as file:
            ds = json.load(file)
            logger.success("JSON file loaded correctly")
            return ds
    except FileNotFoundError:
        logger.error(f"File not found at path: {path}")
        raise FileNotFoundError(f"File not found at path: {path}")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in file: {path}")
        raise json.JSONDecodeError(
            msg=f"Invalid JSON format in file: {path}", doc="", pos=0
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise e


def extract_data_set_info(
    ds: Dict[str, Any],
) -> Tuple[int, List[int], int, np.ndarray]:
    """
    Extracts components from a dataset dictionary.

    Args:
        ds (Dict[str, Any]): The dataset dictionary.

    Returns:
        Tuple[int, List[int], int, np.ndarray]: A tuple containing the number of vertices,
        demands list, vehicle capacity, and distance matrix.

    Raises:
        ValueError: If the dataset is missing required keys or has invalid data.
    """
    if not ds:
        logger.error("Input is not a dictionary!")
        raise ValueError("Input must be a dictionary")

    try:
        num_vertices = int(ds["num_vertices"])
        demands = [int(d) for d in ds["demands"]]
        vehicle_capacity = int(ds["vehicle_capacity"])
        distance_matrix = np.array(ds["distance_matrix"], dtype=np.int64)

        if num_vertices <= 0 or vehicle_capacity <= 0:
            raise ValueError("num_vertices and vehicle_capacity must be positive")
        if len(demands) != num_vertices:
            raise ValueError("Length of demands must match num_vertices")
        if distance_matrix.shape != (num_vertices, num_vertices):
            raise ValueError("Distance matrix dimensions must match num_vertices")

        return num_vertices, demands, vehicle_capacity, distance_matrix
    except KeyError as e:
        missing_key = e.args[0]
        logger.error(f"Missing key in data set: {missing_key}")
        raise ValueError(f"Missing key in data set: {missing_key}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise e
