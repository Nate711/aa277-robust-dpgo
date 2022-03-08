from enum import unique
from typing import List

import jaxlie
from jax import numpy as jnp
import numpy as np
import jaxfg
import copy


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


def partition_graph(poses, factors, agent_node_map):
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

  for agent_i in agent_node_map:
    # print("\n\nAgent I=", agent_i)

    # nodes in agent i
    agent_to_global_map = agent_node_map[agent_i]
    # print("Nodes:", agent_to_global_map)

    global_to_agent_map = invert_list(agent_to_global_map)

    # make local poses
    agent_poses = [copy.copy(poses[i]) for i in agent_to_global_map]
    agent_poses_all[agent_i] = agent_poses
    # print("Agent poses:", agent_poses)

    # make list of internal between factors
    between_factors = []
    for f in factors:
      # get global graph indices of the factor variables
      global_factor_indices = tuple(pose_index_map[v] for v in f.variables)

      # check if the global pose indices are subset of agent global node indices
      if set(global_factor_indices).issubset(set(agent_to_global_map)):
        if type(f) == jaxfg.geometry.BetweenFactor:
          # find local poses corresponding to the global poses
          # print("global factor indices", global_factor_indices)
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
    # print("\nAgent I factors:", between_factors)
    
    # Add priors corresponding to shared nodes
    prior_factors = []
    shared_node_indices = []
    for (agent_j, agent_j_nodes) in agent_node_map.items():
      if agent_i != agent_j:
        agent_j_global_to_local_map = invert_list(agent_j_nodes)
        for shared_node in set(agent_to_global_map).intersection(agent_j_nodes):
          agent_pose = agent_poses[global_to_agent_map[shared_node]]
          prior_factors.append(
              jaxfg.geometry.PriorFactor.make(
                  variable=agent_pose,
                  mu=jaxlie.SE2.from_xy_theta(
                      0.0,
                      0.0,
                      0.0,
                  ),
                  noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
              ))
          shared_node_indices.append(
              (agent_j, shared_node, agent_j_global_to_local_map[shared_node]))
    # print("\nAgent I priors:", prior_factors)
    all_factors = between_factors + prior_factors
    agent_graphs[agent_i] = jaxfg.core.StackedFactorGraph.make(all_factors)
    communication_connections[agent_i] = shared_node_indices
  return agent_graphs, agent_poses_all, communication_connections


if __name__ == "__main__":
  test_partition_graph()