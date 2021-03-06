from enum import unique
from typing import List

import jaxlie
from jax import numpy as jnp
import numpy as np
import jaxfg
import copy
import _laplacian


def test_partition_graph():
  pose_variables: List[jaxfg.geometry.SE2Variable] = [
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
          T_a_b=jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0),
          noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
      ),
      jaxfg.geometry.BetweenFactor.make(
          variable_T_world_a=pose_variables[3],
          variable_T_world_b=pose_variables[0],
          T_a_b=jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0),
          noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
      ),
  ]
  (graphs, poses, comms) = partition_graph(pose_variables,
                                           factors,
                                           agent_node_map={
                                               'agent A': [0, 1, 2],
                                               'agent B': [1, 2, 3],
                                           })
  # TODO, more tests
  assert len(poses['agent A']) == 3
  assert len(poses['agent B']) == 3
  assert comms == {
      'agent A': [('agent B', 1, 0), ('agent B', 2, 1)],
      'agent B': [('agent A', 1, 1), ('agent A', 2, 2)]
  }
  print("GRAPH PARTITION TEST PASSED")


def invert_map(unique_map):
  return {v: k for (k, v) in unique_map.items()}


def invert_list(unique_list):
  return invert_map({i: v for (i, v) in enumerate(unique_list)})


def partition_graph(poses,
                    factors,
                    agent_node_map,
                    prior_precision,
                    noise_model,
                    huber_delta=1.0,
                    use_onp=True):
  """
  Args:
    poses: List of pose variables in global graph
    factors: List of factors in global graph
    agent_node_map: Map from agent name to which nodes in global graph it has
  
  Returns:
    agent_graphs: Map from agent to its graph
    agnet_poses_all: Map from agent to its pose variables
    communication_connections: Map from agent to list. The ith entry in the list 
      corresponds to ith prior factor. The ith entry is a tuple of (which other 
      agent the prior factor is associated with, the global id of the shared node, 
      the id of the node in the other agent's graph). To update the priors based on
      the optimization result. Loop through each agent's list. For each element in 
      agent i's list, set the corresponding prior factor for agent i according to
      the optimized value from its connected agent, agent j, using the tuple entries.
  """
  # maps from pose variable to index in pose list
  pose_index_map = invert_list(poses)

  agent_graphs = {}
  agent_poses_all = {}
  communication_connections = {}
  global_node_to_solution_order = {}

  for agent_i in agent_node_map:
    # nodes in agent i
    agent_to_global_map = agent_node_map[agent_i]
    global_to_agent_map = invert_list(agent_to_global_map)

    # make local poses
    agent_poses = [copy.copy(poses[i]) for i in agent_to_global_map]
    agent_poses_all[agent_i] = agent_poses

    # Track order in which agent poses are entered into between factors
    # index i corresponds to ith row in solution variable, value is global node index
    agent_global_nodes_between_order = []

    # make list of internal between factors
    between_factors = []
    for f in factors:
      # get global graph indices of the factor variables
      global_factor_indices = tuple(pose_index_map[v] for v in f.variables)

      # check if the global pose indices are subset of agent global node indices
      if set(global_factor_indices).issubset(set(agent_to_global_map)):
        if type(f) == jaxfg.geometry.BetweenFactor:

          # Add factor to list if we haven't seen it before
          if global_factor_indices[0] not in agent_global_nodes_between_order:
            agent_global_nodes_between_order.append(global_factor_indices[0])
          if global_factor_indices[1] not in agent_global_nodes_between_order:
            agent_global_nodes_between_order.append(global_factor_indices[1])

          # find local poses corresponding to the global poses
          agent_pose_indices = tuple(
              global_to_agent_map[global_factor_indices[i]] for i in (0, 1))
          factor_poses = tuple(agent_poses[i] for i in agent_pose_indices)

          # add the connection between them
          between_factors.append(
              jaxfg.geometry.BetweenFactor.make(
                  variable_T_world_a=factor_poses[0],
                  variable_T_world_b=factor_poses[1],
                  T_a_b=copy.copy(f.T_a_b),
                  noise_model=copy.copy(f.noise_model)))

    global_node_to_solution_order[agent_i] = invert_list(
        agent_global_nodes_between_order)
    prior_factors = []
    shared_node_indices = []
    for (agent_j, agent_j_nodes) in agent_node_map.items():
      if agent_i != agent_j:
        agent_j_global_to_local_map = invert_list(agent_j_nodes)
        for shared_node in set(agent_to_global_map).intersection(agent_j_nodes):
          agent_pose = agent_poses[global_to_agent_map[shared_node]]
          
          # handle SE2 or SE3
          if type(agent_pose) == jaxfg.geometry._lie_variables.SE2Variable:
            mu = jaxlie.SE2.identity()
            noise_elements = 3
          elif type(agent_pose) == jaxfg.geometry._lie_variables.SE3Variable:
            mu = jaxlie.SE3.identity()
            noise_elements = 6
          else:
            raise NotImplementedError

          # Noise model for prior enforcing agent consensus
          noise_model_ = jaxfg.noises.DiagonalGaussian(jnp.ones(noise_elements) * prior_precision)
          
          if noise_model == 'huber':
            noise_model_ = jaxfg.noises.HuberWrapper(noise_model_, delta=huber_delta)
          elif noise_model == 'laplacian':
            noise_model_ = _laplacian.LaplacianWrapper(noise_model_)
          

          prior_factors.append(
              jaxfg.geometry.PriorFactor.make(
                  variable=agent_pose,
                  mu=mu,
                  noise_model=noise_model_
              ))
          shared_node_indices.append(
              [agent_j, shared_node, agent_j_global_to_local_map[shared_node]])
    all_factors = between_factors + prior_factors
    agent_graphs[agent_i] = jaxfg.core.StackedFactorGraph.make(all_factors,
                                                               use_onp=use_onp)
    communication_connections[agent_i] = shared_node_indices
  for agent in agent_node_map:
    for shared_node_tup in communication_connections[agent]:
      (other_agent, global_node, _) = shared_node_tup
      shared_node_tup[2] = global_node_to_solution_order[other_agent][
          global_node]

  return agent_graphs, agent_poses_all, communication_connections


if __name__ == "__main__":
  test_partition_graph()