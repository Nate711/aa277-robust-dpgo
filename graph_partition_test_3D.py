from typing import List

import jaxlie
from jax import numpy as jnp
import jaxfg
import jax
import graph_tools_nathan
import _g2o_utils_dist

OUTER_LOOP_ITERS = 20
FIXED_FACTOR_EPS = 1.0e3
USE_ONP = True

initial_sqrt_precision = 1.0
prior_precision = 1.0
gamma = 0.75

# HEXAGON EXAMPLE
# pose_variables: List[jaxfg.geometry.SE2Variable] = [
#     jaxfg.geometry.SE2Variable(),
#     jaxfg.geometry.SE2Variable(),
#     jaxfg.geometry.SE2Variable(),
#     jaxfg.geometry.SE2Variable(),
#     jaxfg.geometry.SE2Variable(),
#     jaxfg.geometry.SE2Variable(),
# ]
# factors: List[jaxfg.core.FactorBase] = [
#     jaxfg.geometry.BetweenFactor.make(
#         variable_T_world_a=pose_variables[0],
#         variable_T_world_b=pose_variables[1],
#         T_a_b=jaxlie.SE2.from_xy_theta(1, 0, 0.0),
#         noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
#     ),
#     jaxfg.geometry.BetweenFactor.make(
#         variable_T_world_a=pose_variables[1],
#         variable_T_world_b=pose_variables[2],
#         T_a_b=jaxlie.SE2.from_xy_theta(0, 1.0, 0.0),
#         noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
#     ),
#     jaxfg.geometry.BetweenFactor.make(
#         variable_T_world_a=pose_variables[3],
#         variable_T_world_b=pose_variables[2],
#         T_a_b=jaxlie.SE2.from_xy_theta(1.0, 0, 0.0),
#         noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
#     ),
#     jaxfg.geometry.BetweenFactor.make(
#         variable_T_world_a=pose_variables[3],
#         variable_T_world_b=pose_variables[4],
#         T_a_b=jaxlie.SE2.from_xy_theta(-1.0, 0.0, 0.0),
#         noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
#     ),
#     jaxfg.geometry.BetweenFactor.make(
#         variable_T_world_a=pose_variables[5],
#         variable_T_world_b=pose_variables[4],
#         T_a_b=jaxlie.SE2.from_xy_theta(0.0, 1.0, 0.0),
#         noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
#     ),
#     jaxfg.geometry.BetweenFactor.make(
#         variable_T_world_a=pose_variables[5],
#         variable_T_world_b=pose_variables[0],
#         T_a_b=jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0),
#         noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
#     ),
# ]
# (graphs, poses, comms) = graph_tools_nathan.partition_graph(
#     pose_variables,
#     factors,
#     agent_node_map={
#         'agent A': [0, 2, 1],
#         'agent B': [2, 3, 4],
#         'agent C': [4, 0, 5],
#     },
#     prior_precision=initial_sqrt_precision,
#     noise_model='gaussian',
#     use_onp=USE_ONP)

#### SQUARE EXAMPLE ###
# g2o_path = 'data/square_g2o.g2o'
# g2o, pose_variables, agent_node_list = _g2o_utils_dist.parse_g2o(g2o_path)

# (graphs, poses, comms) = graph_tools_nathan.partition_graph(
#     pose_variables,
#     g2o.factors,
#     agent_node_map=agent_node_list,
#     prior_precision=initial_sqrt_precision,
#     noise_model='gaussian',
#     use_onp=USE_ONP)

##### SPHERE EXAMPLE ##########
g2o_path = 'data/sphere2500.g2o'
g2o, pose_variables, _ = _g2o_utils_dist.parse_g2o(g2o_path)

# agent_node_list = {'A':list(range(0,1501)),'B':[1500]}#list(range(1500,1500
agent_node_list = {
    'A': list(range(0, 1000 + 5)),
    'B': list(range(1000, 1500 + 5)),
    'C': list(range(1500, 2500))
}

# agent_node_list = {
#     'A': list(range(0, 10 + 1)),
#     'B': list(range(10, 15 + 1)),
#     'C': list(range(15, 25))
# }

(graphs, poses, comms) = graph_tools_nathan.partition_graph(
    pose_variables,
    g2o.factors,
    agent_node_map=agent_node_list,
    prior_precision=initial_sqrt_precision,
    noise_model='gaussian',
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
    print("Outer Iter:", i)
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

        for i in range(prior_factor_mu.shape[0]):
            (agent_id, global_node_id, agent_node_id) = comms[agent][i]
            # are they put in in order how appearance in factors??
            
            if var_type == jaxfg.geometry.SE2Variable:
                node_solution = graph_solutions[-1][agent_id].get_stacked_value(
                    var_type).unit_complex_xy[agent_node_id, :]
            else:
                node_solution = graph_solutions[-1][agent_id].get_stacked_value(
                    var_type).wxyz_xyz[agent_node_id, :]
            new_factor_mu = prior_factor_mu[i, :] * (
                1 - gamma) + node_solution * gamma
            # jax version
            if (USE_ONP):
                prior_factor_mu[i, :] = new_factor_mu
            else:
                prior_factor_mu.at[i, :].set(
                    new_factor_mu)

import matplotlib.pyplot as plt

# var_type = type(poses[poses.keys()[0]][0])
# first_agent = 
var_type = next(iter(pose_variables))

alpha = 1.0
if isinstance(var_type, jaxfg.geometry.SE3Variable):
    ax = plt.axes(projection="3d")
    ax.set_box_aspect((1, 1, 1))

for i, multi_agent_solution in enumerate(graph_solutions):
    for agent_id in multi_agent_solution:
        if isinstance(var_type, jaxfg.geometry.SE2Variable):
            data = (multi_agent_solution[agent_id].get_stacked_value(
                jaxfg.geometry.SE2Variable).translation().T)
            # print(data)
            plt.plot(
                *data,
                alpha=1 - alpha,
                c={
                    'agent A': 'r',
                    'agent B': 'g',
                    'agent C': 'b',
                    '0': 'k',
                    '1': 'r',
                    '2': 'g',
                    '3': 'b',
                }[agent_id],
                label=f'{agent_id}, iter={i} ' if i == len(graph_solutions) -
                1 else "_",
            )
        elif isinstance(var_type, jaxfg.geometry.SE3Variable):
            ax.plot3D(
                *(
                    multi_agent_solution[agent_id].get_stacked_value(jaxfg.geometry.SE3Variable)
                    .translation()
                    .T
                ),
                c={'A':'r','B':'g','C':'b'}[agent_id],
                label="Optimized",
            )
    alpha *= 0.9
    # plt.legend()
plt.savefig('hi.png')
plt.show()