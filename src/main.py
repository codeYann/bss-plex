from loguru import logger
from formulations.bss_mtz import Mtz
from formulations.bss_flow import Flow
from utils.data_set import extract_data_set_info, get_data_json
from utils.matrix import fix_diagonal_weight
import threading

if __name__ == "__main__":
    try:
        ds = get_data_json("../../data/Bari/3Bari10.json")
        (num_vertices, demands, vehicle_capacity, distance_matrix) = (
            extract_data_set_info(ds)
        )

        fix_diagonal_weight(distance_matrix)

        V = set(range(num_vertices))
        A = {}

        for i in V:
            for j in V:
                A[(i, j)] = distance_matrix[i, j]

        m = 1
        Q = vehicle_capacity
        q = demands
        c = distance_matrix
        # flow = Flow(V, A, m, Q, q, c)
        mtz = Mtz(V, A, m, Q, q, c)

        threads = [
            # threading.Thread(target=flow.solve, args=["../logs/flow.log"]),
            threading.Thread(target=mtz.solve, args=["../logs/mtz.log"]),
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    except Exception as error:
        raise error
