from utils.data_set import get_data_json, extract_data_set_info
from utils.matrix import fix_diagonal_weight
from itertools import product
from mip import CutType
import networkx as nx
import mip


def initialize_bss_components(
    path: str,
):
    data_set = get_data_json(path)
    (num_vertices, demands, vehicle_capacity, distance_matrix) = extract_data_set_info(
        data_set
    )

    fix_diagonal_weight(distance_matrix)

    V = set(range(num_vertices))
    A = {(i, j): distance_matrix[i, j] for i in V for j in V}
    m = 4
    Q = vehicle_capacity
    q = demands
    c = distance_matrix
    return (V, A, m, Q, q, c)


if __name__ == "__main__":
    try:
        (V, A, m, Q, q, c) = initialize_bss_components(
            "../../data/ReggioEmilia/4ReggioEmilia30.json"
        )

        model = mip.Model()

        # Setting variables
        x = [
            [model.add_var(name=f"x_{i}_{j}", var_type=mip.BINARY) for i in V]
            for j in V
        ]

        f = [
            [model.add_var(name=f"f_{i}_{j}", var_type=mip.CONTINUOUS) for i in V]
            for j in V
        ]

        # objective function: minimize the distance
        model.objective = mip.minimize(
            mip.xsum(c[i, j] * x[i][j] for i in V for j in V)
        )

        # Adding constraints
        model += mip.xsum(x[i][i] for i in V) == 0

        for i in V:
            model += mip.xsum(x[i][j] for j in V - {0}) == 1
            model += mip.xsum(x[j][i] for j in V - {0}) == 1

        model += mip.xsum(x[0][j] for j in V) <= m

        model += (
            mip.xsum(x[0][j] for j in V - {0}) - mip.xsum(x[j][0] for j in V - {0})
        ) == 0

        for j in V - {0}:
            model += mip.xsum(f[j][i] for i in V) - mip.xsum(f[i][j] for i in V) == q[j]

        for i, j in A:
            model += max(0, q[i], -q[j]) * x[i][j] <= f[i][j]
            model += f[i][j] <= min(Q, Q + q[i], Q - q[j]) * x[i][j]

        new_contraints = True
        while new_contraints:
            model.optimize(relax=True)
            print(
                "status: {} objective value : {}".format(
                    model.status, model.objective_value
                )
            )

            G = nx.DiGraph()
            for i, j in A:
                G.add_edge(i, j, capacity=x[i][j].x)

            new_contraints = False
            for u, v in [(i, j) for (i, j) in product(V, V) if i != j]:
                cut_value, (S, NS) = nx.minimum_cut(G, u, v)
                if cut_value <= 0.99 and 0 not in S:
                    model += (
                        mip.xsum(
                            x[i][j] for i, j in A if i in S and j in S and len(S) != 0
                        )
                        <= len(S) - 1
                    )
                    new_contraints = True
            if not new_contraints and model.solver_name.lower() == "cbc":
                cp = model.generate_cuts(
                    [
                        CutType.GOMORY,
                        CutType.MIR,
                        CutType.ZERO_HALF,
                        CutType.KNAPSACK_COVER,
                    ]
                )

                if cp.cuts:
                    model += cp
                    new_contraints = True

        if model.num_solutions:
            route = [(i, j) for i in V for j in V if x[i][j].x >= 0.5]

            flow = [(i, j) for i in V for j in V if f[i][j].x]

            for i, j in route:
                print(f"{model.var_by_name(f"x_{i}_{j}")} = {x[i][j].x}")

            for i, j in flow:
                print(f"{model.var_by_name(f"f_{i}_{j}")} = {f[i][j].x}")

    except Exception as e:
        raise e
