# Code adapted from Schneider et al. https://github.com/RealVNF/distributed-drl-coordination

from env.environment import NetworkEnv
import gym
from spr_rl.params import Params
from gym.spaces import Discrete, Box
from gym.utils import seeding
from spr_rl.envs.wrapper import SPRSimWrapper
import numpy as np
import random
import csv
import simpy
from sprinterface.action import SPRAction
from sprinterface.state import SPRState


def clock(env, name, tick, spr_env, sim_state):
    while True:
        # print(name, env.now)
        yield env.timeout(tick)
        current_time = env.now
        spr_env.timestamps.append(current_time)

class SprEnv(NetworkEnv):
    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.

    The main API methods that users of this class need to know are:

        step
        reset
        render
        close
        seed

    And set the following attributes:

        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards

    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.

    The methods are accessed publicly as "step", "reset", etc...
    """
    # Set this in SOME subclasses
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        self.params: Params = kwargs['params']
        # Obs space is currently temporary
        self.action_space = Discrete(self.params.action_limit)
        # Upper bound for the obs here is 1000, I assume TTL no higher than 1000
        self.observation_space = Box(-1, 1000, shape=self.params.observation_shape)
        self.wrapper = SPRSimWrapper(self.params)
        self.process = None
        self.timestamps = []
        self.spr_state = None
        self.node_state = None
        self.node_obs_ts = []
                
        self.debug =[]
        self.nodej =[]
        if not self.params.test_mode:
            file_hash = random.randint(0, 99999999)
            file_name = f"episode_reward_{file_hash}.csv"
            self.episode_reward_stream = open(f"{self.params.result_dir}/{file_name}", 'a+', newline='')
            self.episode_reward_writer = csv.writer(self.episode_reward_stream)
            self.episode_reward_writer.writerow(['episode', 'reward'])
        self.episode_number = -1  # -1 here so that first episode reset call makes it 0

    def debug_func(self, sim_state: SPRState):

        """
        Gets the network state (local and partial observation of the agent and the neighbour nodes)

        Returns:
            the node observation from current timestamp
        """
        self.flow = sim_state.flow
        self.sfcs = sim_state.sfcs
        self.network = sim_state.network

        # get neighbor nodes
        neighbor_node_ids = list(sim_state.network[self.flow.current_node_id].keys())

        self.node_and_neighbors = [self.flow.current_node_id]

        self.node_and_neighbors.extend([node_id for node_id in neighbor_node_ids])

        # for i in (self.node_and_neighbors):
        #     print(self.node_and_neighbors)
        # for i, node_id in enumerate(self.node_and_neighbors):
        #     print(i, node_id)
        # for node in self.network:
        #     print("asd")
        #     print(self.network["pop5"])
        #     print("start")
        #     print(node)
        #     print("end")
        #     print(self.network[node])

        # return self.node_and_neighbors

    def _node_observation(self, node):

        """
        Gets the network state (local and partial observation of the agent and the neighbour nodes)

        Returns:
            the node observation from current timestamp for each node
        """
        # TODO: Observations have to be adapted..
        self.node = node

        # get neighbor nodes
        neighbor_node_ids = list(self.network[node].keys())

        self.node_and_neighbors = [node]

        self.node_and_neighbors.extend([node_id for node_id in neighbor_node_ids])

        # remaining_node_resources = np.full((self.params.node_resources_size, ), -1.0, dtype=np.float32)

        # for i, node_id in enumerate(self.node_and_neighbors):
        #     node_remaining_cap = self.network.nodes[node_id]['remaining_cap']

        #     current_sf = self.flow.current_sf
        #     if self.flow.forward_to_eg:
        #         current_sf = self.sfcs[self.flow.sfc][-1]
        #     resource_function = self.wrapper.simulator.params.sf_list[current_sf]['resource_function']
        #     if not self.flow.forward_to_eg:
        #         requested_resources = resource_function(self.flow.dr)
        #     else:
        #         requested_resources = 0
        #     node_remaining_cap_norm = (node_remaining_cap - requested_resources) / self.params.max_node_cap

        #     node_remaining_cap_norm = np.clip(node_remaining_cap_norm, -1.0, 1.0)
        #     remaining_node_resources[i] = node_remaining_cap_norm

        # remaining_link_resources = np.full((self.params.link_resources_size, ), -1.0, dtype=np.float32)
        # for i, node_id in enumerate(neighbor_node_ids):
        #     link_remaining_cap = self.network[node][node_id]['remaining_cap']

        #     link_remaining_cap_norm = (link_remaining_cap - self.flow.dr) / self.params.max_link_caps[
        #         node]
        #     link_remaining_cap_norm = np.clip(link_remaining_cap_norm, -1.0, 1.0)

        #     remaining_link_resources[i] = link_remaining_cap_norm

        # # If neighbor does not exist, set distance to -1
        # neighbors_dist_to_eg = np.full((self.params.neighbor_dist_to_eg, ), -1.0, dtype=np.float32)
        # for i, node_id in enumerate(neighbor_node_ids):
        #     if self.flow.egress_node_id is not None:
        #         # Check whether distance from current node to neighbor node should also be included
        #         dist_to_node = self.network.graph['shortest_paths'][(node,
        #                                                              node_id)][1]
        #         dist_to_eg = dist_to_node + self.network.graph['shortest_paths'][(
        #             node_id,
        #             self.flow.egress_node_id)][1]
        #         neighbors_dist_to_eg[i] = (self.flow.ttl - dist_to_eg) / self.flow.ttl
        #     else:
        #         neighbors_dist_to_eg[i] = -1

        # Component availability status
        # vnf_availability = np.full((self.params.vnf_status, ), -1.0, dtype=np.float32)
        # for i, node_id in enumerate(self.node_and_neighbors):
        #     flow_sf = self.flow.current_sf
        #     if flow_sf in self.network.nodes[node_id]['available_sf']:
        #         vnf_availability[i] = 1
        #     else:
        #         vnf_availability[i] = 0

        # TODO: Add actual node observation
        node_observation = np.concatenate(
            (
                # flow_proc_percentage,
                # ttl,
                # vnf_availability,   # not for each flow but for all of them
                # remaining_node_resources,
                # remaining_link_resources,
                # neighbors_dist_to_eg
                np.array([[1, 2, 3]])  # dummy data
            ),
            axis=0
        )
        return node_observation

    def get_node_observation(self):

        """
        Gets the node and each sim_state

        Returns:
            the node observation from current timestamp for all nodes
        """

        # self.network = sim_state.network
        node_obs = []
        for node in self.network:
            node_obs.append(self._node_observation(node))

        return np.array(node_obs)

    def get_netmon_rounds_in_step(self):
        return len(self.spr_env.timestamps)

    def get_num_nodes(self):
        """
        Get number of nodes in the environment.

        :return: number of nodes
        """
        return len(self.network)

    def get_num_agents(self):
        """
        Get number of agents in the environment.

        :return: number of agents
        """
        return 1

    def get_nodes_adjacency(self):
        """
        Gets the node and each sim_state

        Returns:
            the node observation from current timestamp for all nodes
        """

        node_adjacency = []
        # node_adjacency = np.zeros((len(self.network), len(self.network)))

        for node in self.network:
            # i = self._get_nodes_adjacency(node, sim_state)[0][0]
            node_adjacency.append(self._get_nodes_adjacency(node))

        node_adjacency = np.concatenate(node_adjacency, axis=0)
        return node_adjacency

    def _get_node_id_int(self, node_id_str):
        if node_id_str == 0:
            return 0
        return int(node_id_str[3:])

    def _get_nodes_adjacency(self, node) -> np.ndarray:
        """
        Get a matrix of shape (n_nodes, n_nodes) that indicates node adjacency

        :return: node adjacency matrix
        """
        adjacency = np.zeros((1, len(self.network)))
        
        # get neighbor nodes
        neighbor_node_ids = list(self.network[node].keys())
        self.node_and_neighbors = [node]
        self.node_and_neighbors.extend([node_id for node_id in neighbor_node_ids])
        s1=[]
        for node in self.node_and_neighbors:
            s1.append(self._get_node_id_int(node))

        for i in s1:
            adjacency[0][i] = 1
        return adjacency

    def get_node_agent_matrix(self) -> np.ndarray:
        """
        Get a matrix that indicates where agents are located,
        matrix[n, a] = 1 if agent a is on node n and 0 otherwise.

        :return: the node agent matrix of shape (n_nodes, n_agents)
        """
        agent_pos = np.zeros((len(self.network), 1), dtype=int)
        agent_pos[self._get_node_id_int(self.flow.current_node_id)] = 1
        return agent_pos

    def get_node_aux(self):
        """
        Optional auxiliary targets for each node in the network.

        :return: None (default) or auxiliary targets of shape (num_nodes, node_aux_target_size)
        """
        return None

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        action = action[0]
        
        # Get flow information before action
        processing_index = self.last_flow.processing_index
        forward_to_eg = self.last_flow.forward_to_eg
        previous_node_id = self.last_flow.current_node_id
        flow_delay = self.last_flow.end2end_delay

        # Apply action
        nn_state, sim_state = self.wrapper.apply(action)
        new_flow = sim_state.flow
        self.flow = new_flow

        sfc_len = len(sim_state.sfcs[self.last_flow.sfc])

        # Set reward points
        SUCCESS = 10
        PROCESSED = 1 / sfc_len
        EG_MOVED = -(self.last_flow.end2end_delay - flow_delay) / self.params.net_diameter
        EG_KEPT = -1 / self.params.net_diameter
        DROPPED = -10
        MOVED = 0

        # This reward works by using the concept of aliasing and tracking the flow object in memory
        if self.last_flow.success:
            # If flow successful
            reward = SUCCESS
        else:
            if self.last_flow.dropped:
                # If the flow was dropped
                reward = DROPPED
            else:
                if forward_to_eg:
                    if self.last_flow.current_node_id == self.last_flow.egress_node_id:
                        # Flow arrived at egress, wont ask for more decisions
                        reward = SUCCESS
                    else:
                        if self.last_flow.current_node_id == previous_node_id:
                            # Flow stayed at the node
                            reward = EG_KEPT
                        else:
                            # Flow moved
                            reward = EG_MOVED
                else:
                    # Flow is still processing
                    # if flow processed more
                    if self.last_flow.processing_index > processing_index:
                        if (
                            self.last_flow.current_node_id == self.last_flow.egress_node_id
                        ) and (
                            self.last_flow.processing_index == sfc_len
                        ):
                            # Flow was processed at last sf at egress node,
                            # but success wont be triggered as it will automatically depart
                            reward = SUCCESS
                        else:
                            reward = PROCESSED
                    else:
                        reward = MOVED

        done = False
        # Episode length is a set number of flows
        self.episode_reward += reward
        if not self.params.test_mode and self.wrapper.simulator.env.now >= self.params.episode_length:
            done = True
            self.episode_reward_writer.writerow([self.episode_number, self.episode_reward])
        self.steps += 1

        # Set last flow to new flow. New actions will be generated for the new flow
        self.last_flow = new_flow

        # print(self.wrapper.simulator.env.now, self.wrapper.simulator.id)
        # for i in range(len(self.timestamps)):
        #     self.node_obs_ts.append(self._node_observation(action))
        # print(self.node_obs_ts)
        # print(len(self.node_obs_ts[3]))
        # print(self.timestamps)
        # print(self.nodee)
        # print(self.nodee[0])
        # print(sim_state)
        # print(self.node_observation(sim_state))

        self.timestamps.clear()
        return nn_state[np.newaxis], self.get_agent_adjacency(), np.array([reward], dtype=float), done, {'sim_time': self.wrapper.simulator.env.now}

    def get_agent_adjacency(self):
        return np.array([1], dtype=np.int32)

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns:
            observation (object): the initial observation.
        """

        if self.params.sim_seed is None:
            sim_seed = self.random_gen.randint(0, np.iinfo(np.int32).max, dtype=np.int32)
        else:
            sim_seed = self.params.sim_seed
        nn_state, sim_state = self.wrapper.init(sim_seed)
        
        self.flow = sim_state.flow
        self.spr_state = sim_state
        self.node_state = self.spr_state.network
        self.steps = 0
        self.episode_reward = 0
        self.episode_number += 1

        self.last_flow = sim_state.flow
        self.network = sim_state.network

        if self.process is not None:
            self.process.interrupt()
        self.process = self.wrapper.simulator.env.process(clock(self.wrapper.simulator.env, 'ENV RESET', 0.05, self, self.spr_state))

        return nn_state[np.newaxis], self.get_agent_adjacency()

    @staticmethod
    def get_dist_to_eg(network, flow):
        """ Returns the distance to egress node in hops """
        dist_to_egress = network.graph['shortest_paths'][(flow.current_node_id,
                                                          flow.egress_node_id)][1]  # 1: delay; 2: hops
        return dist_to_egress

    def render(self, mode='cli'):
        assert mode in ['human']

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.random_gen, seed = seeding.np_random()
        return [seed]

    # def _my_callback(self):
    #     # env.run(until=env.timeout(1))
    #     print( "before", self.wrapper.simulator.env.now)
    #     yield self.wrapper.simulator.env.timeout(50)
    #     print("after", self.wrapper.simulator.env.now)
