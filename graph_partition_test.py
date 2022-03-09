from typing import List

import jaxlie
from jax import numpy as jnp
import jaxfg
import jax
import graph_tools

OUTER_LOOP_ITERS = 5
FIXED_FACTOR_EPS = 1.0e3

USE_ONP = True

initial_sqrt_precision = 1.0
gamma = 0.75
pose_variables: List[jaxfg.geometry.SE2Variable] = [
    jaxfg.geometry.SE2Variable(),
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
        T_a_b=jaxlie.SE2.from_xy_theta(1, 0, 0.0),
        noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
    ),
    jaxfg.geometry.BetweenFactor.make(
        variable_T_world_a=pose_variables[1],
        variable_T_world_b=pose_variables[2],
        T_a_b=jaxlie.SE2.from_xy_theta(0, 1.0, 0.0),
        noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
    ),
    jaxfg.geometry.BetweenFactor.make(
        variable_T_world_a=pose_variables[2],
        variable_T_world_b=pose_variables[3],
        T_a_b=jaxlie.SE2.from_xy_theta(-1.0, 0.0, 0.0),
        noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
    ),
    jaxfg.geometry.BetweenFactor.make(
        variable_T_world_a=pose_variables[3],
        variable_T_world_b=pose_variables[4],
        T_a_b=jaxlie.SE2.from_xy_theta(-1.0, 0.0, 0.0),
        noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
    ),
    jaxfg.geometry.BetweenFactor.make(
        variable_T_world_a=pose_variables[4],
        variable_T_world_b=pose_variables[5],
        T_a_b=jaxlie.SE2.from_xy_theta(0.0, -1.0, 0.0),
        noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
    ),
    jaxfg.geometry.BetweenFactor.make(
        variable_T_world_a=pose_variables[5],
        variable_T_world_b=pose_variables[0],
        T_a_b=jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0),
        noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
    ),
]

(graphs, poses,
 comms) = graph_tools.partition_graph(pose_variables,
                                      factors,
                                      agent_node_map={
                                          'agent A': [0, 1, 2, 3],
                                          'agent B': [3, 4, 5, 0],
                                      },
                                      prior_precision=initial_sqrt_precision,
                                      use_onp=USE_ONP)
# Comms indexing
# comms[some agent][index i corresponding to prior i] -> tuple (other agent, index of the node in global graph, index of node in other agent graph)


def big_function(prior_precision=1.0):
    initial_assignments = {
        key: jaxfg.core.VariableAssignments.make_from_defaults(poses[key])
        for key in poses
    }

    # Create list to track graph solutions over outer loop iterations.
    # List[i] = {agent id -> solution}
    graph_solutions = []

    solver_class = jaxfg.solvers.FixedIterationGaussNewtonSolver
    solver_args = {'iterations': 2}
    # solver_class = jaxfg.solvers.GaussNewtonSolver

    for agent in graphs:
        prior_factor_stack = graphs[agent].factor_stacks[1].factor

    # with jaxfg.utils.stopwatch("Outer loop timing"):
    for i in range(OUTER_LOOP_ITERS):
        # TODO: Wrap contents of this foor loop in function
        for agent in graphs:
            prior_factor_stack = graphs[agent].factor_stacks[1].factor
            if USE_ONP:
                prior_factor_stack.noise_model.sqrt_precision_diagonal[:] = jax.numpy.ones_like(
                    prior_factor_stack.noise_model.sqrt_precision_diagonal
                ) * prior_precision
            else:
                prior_factor_stack.noise_model.sqrt_precision_diagonal = jax.numpy.ones_like(
                    prior_factor_stack.noise_model.sqrt_precision_diagonal
                ) * prior_precision

        graph_solutions.append({})
        # solve factor solves
        for agent in graphs:
            graph_solutions[-1][agent] = graphs[agent].solve(
                initial_assignments[agent], solver_class(**solver_args))
        for agent in graphs:
            graph_solutions[-1][agent].storage.block_until_ready()
        initial_assignments[agent] = graph_solutions[-1][agent]

        # Update factors based on solutions
        for agent in graphs:
            for i in range(prior_factor_stack.mu.unit_complex_xy.shape[0]):
                (agent_id, global_node_id, agent_node_id) = comms[agent][i]

                # get solution for prior factor i
                node_solution = graph_solutions[-1][
                    agent_id].get_stacked_value(
                        jaxfg.geometry.SE2Variable).unit_complex_xy[
                            agent_node_id, :]

                # set mu for prior factor i
                # print(i, comms[agent][i], node_solution)
                # factor mus are numpy arrays????
                new_factor_mu = prior_factor_stack.mu.unit_complex_xy[i, :] * (
                    1 - gamma) + node_solution * gamma

                # jax version
                if (USE_ONP):
                    prior_factor_stack.mu.unit_complex_xy[i, :] = new_factor_mu
                else:
                    prior_factor_stack.mu.unit_complex_xy.at[i, :].set(
                        new_factor_mu)

    # agent_A_cost = graphs['agent A'].compute_cost(
    #     graph_solutions[-1]['agent A'])
    # TODO sum over all agent costs
    return sum([
        graphs[agent_id].compute_cost(graph_solutions[-1][agent_id])[0]
        for agent_id in graphs
    ])


big_function(3.0)
# with jaxfg.utils.stopwatch("Outer loop timing"):
# big_function(3.0)
# print("JAX GRAD",jax.grad(big_function)(3.0))
eps = 1e-2
prior_precision = 1.0
lr = 1.0
for i in range(10):
    cost = big_function(prior_precision)
    d_prior_precision = (big_function(prior_precision + eps) - cost) / eps
    print("FINITE DIFF: ", d_prior_precision)
    prior_precision -= d_prior_precision * lr
    print(f'prior_precision={prior_precision}, cost={cost}')

# import matplotlib.pyplot as plt
# alpha = 1.0
# for i, multi_agent_solution in enumerate(graph_solutions):
#     for agent_id in multi_agent_solution:
#         plt.plot(
#             *(
#                 multi_agent_solution[agent_id].get_stacked_value(jaxfg.geometry.SE2Variable)
#                 .translation()
#                 .T
#             ),
#             alpha=1-alpha,
#             c = {'agent A':'r', 'agent B':'g'}[agent_id],
#             label=f'iter={i} {agent_id} - optimized',
#         )
#     alpha *= 0.9
#     plt.legend()
# plt.show()