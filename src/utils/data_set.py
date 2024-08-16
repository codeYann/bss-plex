import json
import numpy as np
from loguru import logger
from typing import Any, Dict, Union, Tuple
import os


def get_data_json(path: str) -> Union[Dict[str, Any], None]:
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
    ds: Union[Dict[str, Any], None],
) -> Tuple[int, np.ndarray, int, np.matrix]:
    """
    Extracts components from a dataset dictionary.

    Returns:
        Tuple[int, np.ndarray, int, np.matrix]: A tuple containing the number of vertices,
        demands array, vehicle capacity, and distance matrix.

    Raises:
        ValueError: If the dataset is None or missing required keys.
    """
    if ds is None:
        logger.error("data set is None!")
        raise ValueError("data set is None!")

    try:
        num_vertices = ds["num_vertices"]
        demands = np.array(ds["demands"])
        vehicle_capacity = ds["vehicle_capacity"]
        distance_matrix = np.matrix(ds["distance_matrix"], dtype=np.int64)

        return num_vertices, demands, vehicle_capacity, distance_matrix
    except KeyError as e:
        missing_key = e.args[0]
        logger.error(f"Missing key in data set: {missing_key}")
        raise ValueError(f"Missing key in data set: {missing_key}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise e
