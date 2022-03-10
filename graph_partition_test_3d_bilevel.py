from typing import List

import jaxlie
from jax import numpy as jnp
import jaxfg
import jax
import graph_tools
import graph_tools_nathan
import matplotlib.pyplot as plt
import os
import _g2o_utils_dist
import scipy

USE_ONP = True

# Outer loop prior optimization params
LEARNING_RATE = 0.1
OUTER_ITERATIONS = 50

# Distributed optimization params
DISTRIBUTED_CONSENSUS_ITERS = 10
FIXED_FACTOR_EPS = 1.0e3
NOISE_MODEL = "gaussian"  # can be "gaussian", "laplacian", "huber"
HUBER_DELTA = 0.1

initial_sqrt_precision = 1.0
gamma = 0.5

graph_solutions = []

##### SPHERE EXAMPLE ##########
g2o_path = 'data/sphere2500.g2o'
g2o, pose_variables, _ = _g2o_utils_dist.parse_g2o(g2o_path)
agent_node_list = {
    'A': list(range(0, 1000 + 5)),
    'B': list(range(1000, 1500 + 5)),
    'C': list(range(1500, 2500))
}
(graphs, poses, comms) = graph_tools_nathan.partition_graph(
    pose_variables,
    g2o.factors,
    agent_node_map=agent_node_list,
    prior_precision=initial_sqrt_precision,
    noise_model='gaussian',
    use_onp=USE_ONP)
# Comms indexing
# comms[some agent][index i corresponding to prior i] -> tuple (other agent, index of the node in global graph, index of node in other agent graph)


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
    for i in range(DISTRIBUTED_CONSENSUS_ITERS):
        # TODO: Wrap contents of this foor loop in function
        for agent in graphs:
            prior_factor_stack = graphs[agent].factor_stacks[1].factor
            if NOISE_MODEL in ['huber', 'laplacian']:
                noise_model = prior_factor_stack.noise_model.wrapped
            elif NOISE_MODEL == 'gaussian':
                noise_model = prior_factor_stack.noise_model
            for prior in range(noise_model.sqrt_precision_diagonal.shape[0]):
                other_agent = comms[agent][prior][0]
                agent_id = int(agent)

                other_agent_id = int(other_agent)
                prior_precisions_softmax = prior_precisions
                # prior_precisions_softmax = scipy.special.softmax(prior_precisions)
                if USE_ONP:
                    noise_model.sqrt_precision_diagonal[
                        prior, :] = jax.numpy.ones_like(
                            noise_model.sqrt_precision_diagonal[prior, :]
                        ) * prior_precisions_softmax[
                            other_agent_id] * prior_precisions_softmax[
                                agent_id] / jnp.mean(prior_precisions_softmax)
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
            pose_type = type(prior_factor_stack.variables[0])
            if pose_type == jaxfg.geometry._lie_variables.SE3Variable:
                prior_factor_mu = prior_factor_stack.mu.wxyz_xyz
                var_type = jaxfg.geometry.SE3Variable
            else:
                prior_factor_mu = prior_factor_stack.mu.unit_complex_xy
                var_type = jaxfg.geometry.SE2Variable
            for i in range(prior_factor_stack.mu.unit_complex_xy.shape[0]):
                (agent_id, global_node_id, agent_node_id) = comms[agent][i]

                # get solution for prior factor i
                if var_type == jaxfg.geometry.SE2Variable:
                    node_solution = graph_solutions[-1][
                        agent_id].get_stacked_value(var_type).unit_complex_xy[
                            agent_node_id, :]
                else:
                    node_solution = graph_solutions[-1][
                        agent_id].get_stacked_value(var_type).wxyz_xyz[
                            agent_node_id, :]

                new_factor_mu = prior_factor_mu[i, :] * (
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
        #])#    + 0.1 * jnp.linalg.norm(scipy.special.softmax(prior_precisions))
    ]) + 1 * (1 - jnp.linalg.norm(prior_precisions))**2


eps = 1e-1
num_agents = len(graphs)
prior_precisions = jnp.ones(num_agents) * initial_sqrt_precision
prior_precisions /= jnp.linalg.norm(prior_precisions)
lr = LEARNING_RATE
costs = []
prior_precisions_log = [prior_precisions]
# prior_precisions_log = [scipy.special.softmax(prior_precisions)]
for gradient_descent_iter in range(OUTER_ITERATIONS):
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
    # prior_precisions_log.append(scipy.special.softmax(prior_precisions))
    prior_precisions_log.append(prior_precisions)

    plt.axis('equal')
    for i, multi_agent_solution in enumerate(graph_solutions):
        for agent_id in multi_agent_solution:
            plt.plot(
                *(multi_agent_solution[agent_id].get_stacked_value(
                    jaxfg.geometry.SE2Variable).translation().T),
                alpha=1 - (i / len(graph_solutions))**2,
                c='C%d' % (int(agent_id)),
                label=f'iter={i} {agent_id} - optimized',
            )
        if NOISE_MODEL == "huber":
            plt.title("noise model: " + NOISE_MODEL + ", Huber delta: " +
                      str(HUBER_DELTA) + " cost: " + str(costs[-1]))
        else:
            plt.title("noise model: " + NOISE_MODEL + " cost: " +
                      str(costs[-1]))
        # plt.legend()
    try:
        os.mkdir(f'anim/{NOISE_MODEL}')
    except FileExistsError:
        pass
    if NOISE_MODEL == "huber":
        plt.savefig(
            f'anim/{NOISE_MODEL}/{str(gradient_descent_iter) + "_Hd_" + str(HUBER_DELTA)}.png'
        )
    else:
        plt.savefig(f'anim/{NOISE_MODEL}/{str(gradient_descent_iter)}.png')
    plt.clf()
    # plt.show()

plt.plot(prior_precisions_log)
plt.legend()
plt.title("Agent confidence versus iterations")
plt.xlabel("Iteration")
plt.ylabel("Agent confidence")
try:
    os.mkdir('weight_plots')
except FileExistsError:
    pass
plt.savefig(
    f'weight_plots/noise={NOISE_MODEL}_huber-delta={HUBER_DELTA}_iters={OUTER_ITERATIONS}_lr={LEARNING_RATE}.png'
)
plt.show()

plt.plot(costs)
plt.legend()
plt.title("Cost versus iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost")
try:
    os.mkdir('cost_plots')
except FileExistsError:
    pass
plt.savefig(
    f'cost_plots/noise={NOISE_MODEL}_huber-delta={HUBER_DELTA}_iters={OUTER_ITERATIONS}_lr={LEARNING_RATE}.png'
)
plt.show()