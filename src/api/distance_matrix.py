from time import sleep
from dotenv import load_dotenv
from typing import Dict, Any, List
import googlemaps as maps
import os
import pandas as pd
import itertools

load_dotenv()

KEY = os.getenv("GOOGLE_KEY")
gmaps = maps.Client(key=KEY)


def calculate_instance_demands(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the demand for each bike station.

    The demand q_i for station i is defined as:
    q_i = B_i^current - B_i^final

    Where:
    - B_i^current is the current number of bikes at station i
    - B_i^final is the desired number of bikes at station i in the final configuration

    In this implementation, the final desired configuration is set to half of the total capacity.

    Args:
        df (pd.DataFrame): DataFrame containing 'available_bikes' and 'free_slots' columns

    Returns:
        pd.Series: Series of demands for each station
    """
    total_capacity = df["available_bikes"] + df["free_slots"]
    desired_bikes = total_capacity // 2
    demands = df["available_bikes"] - desired_bikes
    return demands


def create_bss_instance(df: pd.DataFrame) -> Dict[str, Any]:
    stations = df["station"]
    address = df["address"]
    lat_long = df[["latitude", "longitude"]].apply(tuple, axis=1)
    demands = calculate_instance_demands(df)
    return {
        "stations": stations,
        "address": address,
        "coordinates": lat_long,
        "demands": demands,
    }


def generate_distance_matrix(instance: Dict[str, Any]) -> List[List[int]]:
    coordinates = instance["coordinates"]
    matrix = [[0 for _ in range(len(coordinates))] for _ in range(len(coordinates))]

    chunk_size = 10

    for i, j in itertools.product(range(0, len(coordinates), chunk_size), repeat=2):
        origins = (
            coordinates.iloc[i : i + chunk_size]
            if isinstance(coordinates, pd.Series)
            else coordinates[i : i + chunk_size]
        )
        destinations = (
            coordinates.iloc[j : j + chunk_size]
            if isinstance(coordinates, pd.Series)
            else coordinates[j : j + chunk_size]
        )

        if origins.empty or destinations.empty:
            continue

        result = gmaps.distance_matrix(
            origins.tolist(), destinations.tolist(), mode="driving", units="metric"
        )

        for row_idx, row in enumerate(result["rows"]):
            for col_idx, element in enumerate(row["elements"]):
                matrix[i + row_idx][j + col_idx] = (
                    element["distance"]["value"]
                    if element["status"] == "OK"
                    else 1000000000
                )

        # small delay to avoid api rate limits
        sleep(1.5)

    return matrix
