import cplex


def solve_linear_program():
    # Create a CPLEX problem instance
    problem = cplex.Cplex()

    # Set the problem to be a maximization problem
    problem.set_problem_type(cplex.Cplex.problem_type.MILP)

    # Objective function coefficients
    problem.objective.set_sense(problem.objective.sense.maximize)
    problem.variables.add(obj=[4, 5], names=["x", "y"])

    # Constraints coefficients and bounds
    constraints = [[["x", "y"], [3, 2]], [["x", "y"], [3.0, 1.0]]]
    rhs = [8.0, 9.0]
    senses = [
        "L",
        "L",
    ]  # "L" for less than or equal, "E" for equal, "G" for greater than or equal

    problem.linear_constraints.add(lin_expr=constraints, senses=senses, rhs=rhs)

    # Solve the problem
    problem.solve()

    # Print the results
    print("Solution status:", problem.solution.get_status())
    print("Objective value:", problem.solution.get_objective_value())
    for i, var_name in enumerate(problem.variables.get_names()):
        print(f"{var_name} = {problem.solution.get_values(i)}")


if __name__ == "__main__":
    solve_linear_program()
