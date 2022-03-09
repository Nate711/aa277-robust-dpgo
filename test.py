from typing import List

import jaxlie
from jax import numpy as jnp
import jaxfg

import _laplacian

OUTER_LOOP_ITERS = 80
FIXED_FACTOR_EPS = 1.0e3
gamma = 0.5


def partition_graph(variables, factors, agent_node_map):
    pass

def make_graph_1(fixed_factor_eps=1e6):
    """fixed_factor_eps: square root of precision matrix"""
    pose_variables: List[jaxfg.geometry.SE2Variable] = [
        jaxfg.geometry.SE2Variable(),
        jaxfg.geometry.SE2Variable(),
        jaxfg.geometry.SE2Variable(),
        jaxfg.geometry.SE2Variable(),
        jaxfg.geometry.SE2Variable(),
    ]
    factors: List[jaxfg.core.FactorBase] = [
        # comm
        jaxfg.geometry.BetweenFactor.make(
            variable_T_world_a=pose_variables[0],
            variable_T_world_b=pose_variables[1],
            T_a_b=jaxlie.SE2.from_xy_theta(0, 0, 0.0),
            noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
        ),
        jaxfg.geometry.BetweenFactor.make(
            variable_T_world_a=pose_variables[1],
            variable_T_world_b=pose_variables[2],
            T_a_b=jaxlie.SE2.from_xy_theta(0.0, 1.0, 0.0),
            noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
        ),
        jaxfg.geometry.BetweenFactor.make(
            variable_T_world_a=pose_variables[2],
            variable_T_world_b=pose_variables[3],
            T_a_b=jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0),
            noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
        ),
        # comm
        jaxfg.geometry.BetweenFactor.make(
            variable_T_world_a=pose_variables[3],
            variable_T_world_b=pose_variables[4],
            T_a_b=jaxlie.SE2.from_xy_theta(0, 0, 0.0),
            noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
        ),

        jaxfg.geometry.PriorFactor.make(
            variable=pose_variables[0],
            mu=jaxlie.SE2.from_xy_theta(0.0, -1.1e1, 0.0),
            noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)*fixed_factor_eps),
        ),
        jaxfg.geometry.PriorFactor.make(
            variable=pose_variables[4],
            mu=jaxlie.SE2.from_xy_theta(1.1e1, 0.0, 0.0),
            noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)*fixed_factor_eps),
        ),
        jaxfg.geometry.PriorFactor.make(
            variable=pose_variables[2],
            mu=jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0),
            noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
        ),
        
    ]
    graph = jaxfg.core.StackedFactorGraph.make(factors)
    return pose_variables, graph

def make_graph_2(fixed_factor_eps=1e6):
    """fixed_factor_eps: square root of precision matrix"""
    pose_variables: List[jaxfg.geometry.SE2Variable] = [
        jaxfg.geometry.SE2Variable(),
        jaxfg.geometry.SE2Variable(),
        jaxfg.geometry.SE2Variable(),
        jaxfg.geometry.SE2Variable(),
        jaxfg.geometry.SE2Variable(),
    ]
    factors: List[jaxfg.core.FactorBase] = [
        jaxfg.geometry.BetweenFactor.make(
            variable_T_world_a=pose_variables[0],
            variable_T_world_b=pose_variables[1],
            T_a_b=jaxlie.SE2.from_xy_theta(0, 0, 0.0),
            noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
        ),
        jaxfg.geometry.BetweenFactor.make(
            variable_T_world_a=pose_variables[1],
            variable_T_world_b=pose_variables[2],
            T_a_b=jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0),
            noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
        ),
        jaxfg.geometry.BetweenFactor.make(
            variable_T_world_a=pose_variables[2],
            variable_T_world_b=pose_variables[3],
            T_a_b=jaxlie.SE2.from_xy_theta(0.0, 1.0, 0.0),
            noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
        ),
        jaxfg.geometry.BetweenFactor.make(
            variable_T_world_a=pose_variables[3],
            variable_T_world_b=pose_variables[4],
            T_a_b=jaxlie.SE2.from_xy_theta(0, 0, 0.0),
            noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
        ),
        jaxfg.geometry.PriorFactor.make(
            variable=pose_variables[0],
            mu=jaxlie.SE2.from_xy_theta(-1.1, 0, 0.0),
            noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)*fixed_factor_eps),
        ),
        jaxfg.geometry.PriorFactor.make(
            variable=pose_variables[4],
            mu=jaxlie.SE2.from_xy_theta(0, 1.1, 0.0),
            noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)*fixed_factor_eps),
        ),
    ]
    graph = jaxfg.core.StackedFactorGraph.make(factors)
    return pose_variables, graph

pose_variables_1, graph_1 = make_graph_1(FIXED_FACTOR_EPS)
pose_variables_2, graph_2 = make_graph_2(FIXED_FACTOR_EPS)

initial_assignments_1 = jaxfg.core.VariableAssignments.make_from_defaults(pose_variables_1)
initial_assignments_2 = jaxfg.core.VariableAssignments.make_from_defaults(pose_variables_2)

graph_1_solutions = []
graph_2_solutions = []

# solver = jaxfg.solvers.FixedIterationGaussNewtonSolver(iterations=1)
solver = jaxfg.solvers.GaussNewtonSolver()
solution_assignments_1 = graph_1.solve(initial_assignments_1, solver)
solution_assignments_2 = graph_2.solve(initial_assignments_2, solver)

with jaxfg.utils.stopwatch("Outer loop timing"):

    for i in range(OUTER_LOOP_ITERS):
        # print("Initial assignments:")
        # print(initial_assignments)

        # Solve. Note that the first call to solve() will be much slower than subsequent calls.
        with jaxfg.utils.stopwatch("First solve (slower because of JIT compilation)"):
            solution_assignments_1 = graph_1.solve(initial_assignments_1, solver)
            solution_assignments_1.storage.block_until_ready()  # type: ignore
            solution_assignments_2 = graph_2.solve(initial_assignments_2, solver)
            solution_assignments_2.storage.block_until_ready()  # type: ignore

        graph_1_solutions.append(solution_assignments_1)
        graph_2_solutions.append(solution_assignments_2)

        initial_assignments_1 = solution_assignments_1
        initial_assignments_2 = solution_assignments_2

        # get solutions from communication nodes
        graph_2_node_2 = solution_assignments_2.get_stacked_value(jaxfg.geometry.SE2Variable).unit_complex_xy[1,:]
        graph_2_node_4 = solution_assignments_2.get_stacked_value(jaxfg.geometry.SE2Variable).unit_complex_xy[3,:]
        
        graph_1_node_2 = solution_assignments_1.get_stacked_value(jaxfg.geometry.SE2Variable).unit_complex_xy[1,:]
        graph_1_node_4 = solution_assignments_1.get_stacked_value(jaxfg.geometry.SE2Variable).unit_complex_xy[3,:]
        
        # Update priors
        graph_1.factor_stacks[1].factor.mu.unit_complex_xy[0,:] = graph_1.factor_stacks[1].factor.mu.unit_complex_xy[0,:] * (1 - gamma) + graph_2_node_2 * (gamma)
        graph_1.factor_stacks[1].factor.mu.unit_complex_xy[1,:] = graph_1.factor_stacks[1].factor.mu.unit_complex_xy[1,:] * (1 - gamma) + graph_2_node_4 * (gamma)

        graph_2.factor_stacks[1].factor.mu.unit_complex_xy[0,:] = graph_2.factor_stacks[1].factor.mu.unit_complex_xy[0,:] * (1 - gamma) + graph_1_node_2 * (gamma)
        graph_2.factor_stacks[1].factor.mu.unit_complex_xy[1,:] = graph_2.factor_stacks[1].factor.mu.unit_complex_xy[1,:] * (1 - gamma) + graph_1_node_4 * (gamma)



# Print all solved variable values.
# print("Solutions (jaxfg.core.VariableAssignments):")
# print(solution_assignments_1, solution_assignments_2)
# print()


# Grab and print a single variable value at a time.
# print("First pose (jaxlie.SE2 object):")
# print(solution_assignments.get_value(pose_variables[0]))
# print()



import matplotlib.pyplot as plt
# plt.plot(
#     *(
#         initial_assignments_1.get_stacked_value(jaxfg.geometry.SE2Variable)
#         .translation()
#         .T
#     ),
#     # Equivalent:
#     # *(onp.array([initial_poses.get_value(v).translation() for v in pose_variables]).T),
#     c="r",
#     label="Initial",
# )
# plt.plot(
#     *(
#         initial_assignments_2.get_stacked_value(jaxfg.geometry.SE2Variable)
#         .translation()
#         .T
#     ),
#     # Equivalent:
#     # *(onp.array([initial_poses.get_value(v).translation() for v in pose_variables]).T),
#     c="r",
#     label="Initial",
# )
alpha = 1.0
for (graph_1_soln, graph_2_soln) in zip(graph_1_solutions[-10:-1], graph_2_solutions[-10:-1]):
    plt.plot(
        *(
            graph_1_soln.get_stacked_value(jaxfg.geometry.SE2Variable)
            .translation()
            .T
        ),
        # Equivalent:
        # *(onp.array([solution_poses.get_value(v).translation() for v in pose_variables]).T),
        c="g",
        alpha=1-alpha,
        label="Optimized",
    )
    plt.plot(
        *(
            graph_2_soln.get_stacked_value(jaxfg.geometry.SE2Variable)
            .translation()
            .T
        ),
        # Equivalent:
        # *(onp.array([solution_poses.get_value(v).translation() for v in pose_variables]).T),
        c="r",
        alpha=1-alpha,
        label="Optimized",
    )
    alpha *= 0.9
    plt.legend()
plt.show()