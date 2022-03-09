from typing import List

import jaxlie
from jax import numpy as jnp
import jaxfg
import jax
import graph_tools_nathan

OUTER_LOOP_ITERS = 15
FIXED_FACTOR_EPS = 1.0e3
USE_ONP = True

initial_sqrt_precision = 1.0
prior_precision=1.0
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
        T_a_b=jaxlie.SE2.from_xy_theta(-1.0, 0, 0.0),
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
 comms) = graph_tools_nathan.partition_graph(pose_variables,
                                      factors,
                                      agent_node_map={
                                          'agent A': [0, 2, 1],
                                          'agent B': [2, 3, 4],
                                          'agent C': [4, 0, 5],
                                      },
                                      prior_precision=initial_sqrt_precision,
                                      use_onp=USE_ONP)

# Comms indexing
# comms[some agent][index i corresponding to prior i] -> tuple (other agent, index of the node in global graph, index of node in other agent graph)
print("COMMS:", comms)
initial_assignments = {
    key: jaxfg.core.VariableAssignments.make_from_defaults(poses[key])
    for key in poses
}

# Create list to track graph solutions over outer loop iterations.
# List[i] = {agent id -> solution}
graph_solutions = []

# solver_class = jaxfg.solvers.FixedIterationGaussNewtonSolver
# solver_args = {'iterations': 5}
solver_class = jaxfg.solvers.GaussNewtonSolver
solver_args = {}

# with jaxfg.utils.stopwatch("Outer loop timing"):
for i in range(OUTER_LOOP_ITERS):
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
        
        prior_factor_stack = graphs[agent].factor_stacks[1].factor
        for i in range(prior_factor_stack.mu.unit_complex_xy.shape[0]):
            (agent_id, global_node_id, agent_node_id) = comms[agent][i]
            # are they put in in order how appearance in factors??
            node_solution = graph_solutions[-1][
                agent_id].get_stacked_value(
                    jaxfg.geometry.SE2Variable).unit_complex_xy[
                        agent_node_id, :]
            new_factor_mu = prior_factor_stack.mu.unit_complex_xy[i, :] * (
                1 - gamma) + node_solution * gamma
            # jax version
            if (USE_ONP):
                prior_factor_stack.mu.unit_complex_xy[i, :] = new_factor_mu
            else:
                prior_factor_stack.mu.unit_complex_xy.at[i, :].set(
                    new_factor_mu)

import matplotlib.pyplot as plt
alpha = 1.0
for i, multi_agent_solution in enumerate(graph_solutions):
    for agent_id in multi_agent_solution:
        plt.plot(
            *(
                multi_agent_solution[agent_id].get_stacked_value(jaxfg.geometry.SE2Variable)
                .translation()
                .T
            ),
            alpha=1-alpha,
            c = {'agent A':'r', 'agent B':'g', 'agent C': 'b'}[agent_id],
            label=f'{agent_id}, iter={i} ' if i == len(graph_solutions)-1 else "_",
        )
    alpha *= 0.9
    plt.legend()
plt.savefig('hi.png')