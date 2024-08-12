# from optimization.bss_model import main
from loguru import logger
from optimization.model import BSS
from utils.data_set import extract_data_set_info, get_data_json
from utils.matrix import fix_diagonal_weight


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

        m = 2
        Q = vehicle_capacity
        q = demands
        c = distance_matrix

        print(V, A, m, Q, q, c)

        bss = BSS(V, A, m, Q, q, c)
        bss.solve()
    except Exception as error:
        logger.exception(error)
