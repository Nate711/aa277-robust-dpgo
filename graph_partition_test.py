from typing import List

import jaxlie
from jax import numpy as jnp
import jaxfg
import jax
import graph_tools
import graph_tools_nathan
import matplotlib.pyplot as plt

OUTER_LOOP_ITERS = 5
FIXED_FACTOR_EPS = 1.0e3

USE_ONP = True
NOISE_MODEL = "gaussian"
HUBER_DELTA = 0.1

LEARNING_RATE = 0.1
ITERATIONS = 10

initial_sqrt_precision = 1.0
gamma = 0.5
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
        T_a_b=jaxlie.SE2.from_xy_theta(-1.0, 3, 0.0),
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
                                          'agent A': [0, 1, 2],
                                          'agent B': [2, 3, 4],
                                          'agent C': [4, 5, 0],
                                      },
                                      prior_precision=initial_sqrt_precision,
                                      noise_model=NOISE_MODEL,
                                      huber_delta=HUBER_DELTA,
                                      use_onp=USE_ONP)
# Comms indexing
# comms[some agent][index i corresponding to prior i] -> tuple (other agent, index of the node in global graph, index of node in other agent graph)
graph_solutions = []


def big_function(prior_precisions):
    initial_assignments = {
        key: jaxfg.core.VariableAssignments.make_from_defaults(poses[key])
        for key in poses
    }

    # Create list to track graph solutions over outer loop iterations.
    # List[i] = {agent id -> solution}
    global graph_solutions
    graph_solutions = []

    solver_class = jaxfg.solvers.FixedIterationGaussNewtonSolver
    solver_args = {'iterations': 2}
    # solver_class = jaxfg.solvers.GaussNewtonSolver

    # with jaxfg.utils.stopwatch("Outer loop timing"):
    for i in range(OUTER_LOOP_ITERS):
        # TODO: Wrap contents of this foor loop in function
        for agent in graphs:
            prior_factor_stack = graphs[agent].factor_stacks[1].factor
            if NOISE_MODEL == 'huber':
                noise_model = prior_factor_stack.noise_model.wrapped
            elif NOISE_MODEL == 'gaussian':
                noise_model = prior_factor_stack.noise_model
            for prior in range(noise_model.sqrt_precision_diagonal.shape[0]):
                other_agent = comms[agent][prior][0]
                agent_id = {'agent A': 0, 'agent B': 1, 'agent C': 2}[agent]
                other_agent_id = {
                    'agent A': 0,
                    'agent B': 1,
                    'agent C': 2
                }[other_agent]
                if USE_ONP:
                    noise_model.sqrt_precision_diagonal[
                        prior, :] = jax.numpy.ones_like(
                            noise_model.sqrt_precision_diagonal[prior, :]
                        ) * prior_precisions[agent_id] * prior_precisions[
                            other_agent_id] / jnp.mean(prior_precisions)
                else:
                    noise_model.sqrt_precision_diagonal = jax.numpy.ones_like(
                        noise_model.sqrt_precision_diagonal
                    ) * prior_precisions[other_agent_id]

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
    ]) + 1 * (1 - jnp.linalg.norm(prior_precisions))**2


# big_function(3.0)
# with jaxfg.utils.stopwatch("Outer loop timing"):
# big_function(3.0)
# print("JAX GRAD",jax.grad(big_function)(3.0))
eps = 1e-1
num_agents = len(graphs)
prior_precisions = jnp.ones(num_agents) * initial_sqrt_precision
prior_precisions /= jnp.linalg.norm(prior_precisions)
lr = LEARNING_RATE
costs = []
for gradient_descent_iter in range(ITERATIONS):
    cost = big_function(prior_precisions)
    costs.append(cost)
    d_prior_precisions = jnp.zeros_like(prior_precisions)
    for i in range(num_agents):
        one_hot = jnp.zeros_like(prior_precisions)
        one_hot = one_hot.at[i].set(1)
        d_prior_precisions = d_prior_precisions.at[i].set(
            (big_function(prior_precisions + one_hot * eps) - cost) / eps)
    print("FINITE DIFF: ", d_prior_precisions)
    prior_precisions = prior_precisions - d_prior_precisions * lr
    print(f'prior_precision={prior_precisions}, cost={cost}')

    plt.axis('equal')
    for i, multi_agent_solution in enumerate(graph_solutions):
        for agent_id in multi_agent_solution:
            plt.plot(
                *(multi_agent_solution[agent_id].get_stacked_value(
                    jaxfg.geometry.SE2Variable).translation().T),
                alpha=1 - (i / len(graph_solutions))**2,
                c={
                    'agent A': 'r',
                    'agent B': 'g',
                    'agent C': 'b'
                }[agent_id],
                label=f'iter={i} {agent_id} - optimized',
            )
        # plt.legend()
    plt.savefig(f'anim/{gradient_descent_iter}.png')
    plt.clf()
    # plt.show()
plt.plot(costs)
plt.title("Cost versus iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.savefig(f'cost_plots/noise={NOISE_MODEL}_huber-delta={HUBER_DELTA}_iters={ITERATIONS}_lr={LEARNING_RATE}.png')
plt.show()