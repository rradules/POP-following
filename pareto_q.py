import argparse
import itertools
import gym

import numpy as np

from collections import namedtuple
from pymoo.factory import get_performance_indicator

from utils import mkdir_p, get_non_dominated, get_best, print_pcs, save_momdp, save_pcs, save_training_data


gym.register(
        id='DeepSeaTreasure-v0',
        entry_point='deep_sea_treasure:DeepSeaTreasureEnv')


class ParetoQ:
    """
    An implementation for a pareto Q learning agent that is able to deal with stochastic environments.
    """
    def __init__(self, num_states, num_actions, num_objectives, ref_point, gamma=0.8, epsilon=0.1, decimals=2, novec=30):
        self.num_actions = num_actions
        self.num_states = num_states
        self.num_objectives = num_objectives

        self.gamma = gamma
        self.epsilon = epsilon
        self.decimals = decimals
        self.novec = novec

        self.hv = get_performance_indicator("hv", ref_point=-1*ref_point)  # Pymoo flips everything.

        # Implemented as recommended by Van Moffaert et al. by substituting (s, a) with (s, a, s').
        self.non_dominated = [[[{tuple(np.zeros(num_objectives))} for _ in range(num_states)] for _ in range(num_actions)] for _ in range(num_states)]
        self.avg_r = np.zeros((num_states, num_actions, num_states, num_objectives))
        self.transitions = np.zeros((num_states, num_actions, num_states))

    def calc_q_set(self, state, action):
        """
        This function calculates a complete Q-set for a given state and action.
        :param state: The current state.
        :param action: The selected action.
        :return: A Q(s, a) set.
        """
        q_set = set()

        transition_probs = self.transitions[state, action] / max(1, np.sum(self.transitions[state, action]))
        next_states = np.where(self.transitions[state, action, :] > 0)[0]  # Next states with prob > 0

        next_sets = []
        for next_state in next_states:
            next_sets.append(list(self.non_dominated[state][action][next_state]))

        cartesian_product = itertools.product(*next_sets)

        for next_vectors in cartesian_product:
            expected_vec = np.zeros(self.num_objectives)
            for idx, next_vector in enumerate(next_vectors):
                next_state = next_states[idx]
                transition_prob = transition_probs[next_state]
                disc_future_reward = self.gamma * np.array(next_vector)
                expected_vec += transition_prob * (self.avg_r[state, action, next_state] + disc_future_reward)
            expected_vec = tuple(np.around(expected_vec, decimals=self.decimals))  # Round the future reward.
            q_set.add(tuple(expected_vec))
        return q_set

    def select_action(self, state):
        """
        This function selects an action using epsilon greedy on the hypervolume metric given the current state.
        :param state: The current state.
        :return: The action to take.
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            hypervolumes = []
            for action in range(self.num_actions):
                q_set = self.calc_q_set(state, action)
                q_set = get_non_dominated(q_set)
                hypervolume = self.hv.do(-1*np.array(list(q_set)))
                hypervolumes.append(hypervolume)
            return np.random.choice(np.argwhere(hypervolumes == np.max(hypervolumes)).flatten())

    def update(self, state, action, next_state, r):
        """
        This function updates the agent.
        :param state: The previous state.
        :param action: The action that was selected.
        :param next_state: The state that the agent ended up in.
        :param r: The obtained reward.
        :return: /
        """
        self.transitions[state, action, next_state] += 1
        q_sets = []
        for a in range(self.num_actions):
            q_sets.append(self.calc_q_set(next_state, a))
        self.non_dominated[state][action][next_state] = get_best(set().union(*q_sets), max_points=self.novec)
        self.avg_r[state, action, next_state] += (r - self.avg_r[state, action, next_state])/self.transitions[state, action, next_state]

    def construct_pcs(self):
        """
        This function constructs the final PCS.
        :return: The PCS.
        """
        pcs = [[{tuple(np.zeros(self.num_objectives))} for _ in range(self.num_actions)] for _ in range(self.num_states)]
        for state in range(self.num_states):
            for action in range(self.num_actions):
                pcs[state][action] = get_non_dominated(self.calc_q_set(state, action))
        return pcs

    def construct_nn_dataset(self):
        """
        This function constructs a NN dataset with training samples generated from the learned Q sets.
        :return: The trajectory dataset.
        """
        dataset = []

        for state in range(self.num_states):
            for action in range(self.num_actions):
                transition_probs = self.transitions[state, action] / max(1, np.sum(self.transitions[state, action]))
                next_states = np.where(self.transitions[state, action, :] > 0)[0]  # Next states with prob > 0

                next_sets = []
                for next_state in next_states:
                    next_sets.append(list(self.non_dominated[state][action][next_state]))

                cartesian_product = itertools.product(*next_sets)

                for next_vectors in cartesian_product:
                    N = np.zeros(self.num_objectives)
                    for idx, next_vector in enumerate(next_vectors):
                        next_state = next_states[idx]
                        transition_prob = transition_probs[next_state]
                        N += transition_prob * np.array(next_vector)

                    for next_state, next_vector in zip(next_states, next_vectors):  # Add the trajectory to the dataset.
                        dataset.append(Data(next_vector, N, state, action, next_state))

        return dataset


def run_pql(env, num_iters=100, max_t=20, decimals=3, epsilon=0.1, gamma=0.8, novec=30):
    """
    This function runs PQL.
    :param env: The environment the run PQL in.
    :param num_iters: The number of iterations to run PQL for.
    :param max_t: The maximum timestep per iteration.
    :param decimals: The maximum number of decimals used for rewards.
    :param epsilon: The exploration parameter.
    :param gamma: The discount factor.
    :param novec: The maximum number of vectors per set.
    :return: The PCS and NN dataset.
    """
    agent = ParetoQ(num_states, num_actions, num_objectives, ref_point, gamma=gamma, epsilon=epsilon, decimals=decimals, novec=novec)

    for i in range(num_iters):
        print(f'Performing iteration {i}')
        state = env.reset()
        done = False
        timestep = 0

        while not done and timestep < max_t:
            action = agent.select_action(state)
            next_state, r, done, prob = env.step(action)
            agent.update(state, action, next_state, r)
            state = next_state
            timestep += 1

    pcs = agent.construct_pcs()
    nn_dataset = agent.construct_nn_dataset()

    return pcs, nn_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, default='DeepSeaTreasure-v0', help="The environment to run PVI on.")
    parser.add_argument('-states', type=int, default=10, help="The number of states. Only used with the random MOMDP.")
    parser.add_argument('-obj', type=int, default=2, help="The number of objectives. Only used with the random MOMDP.")
    parser.add_argument('-act', type=int, default=2, help="The number of actions. Only used with the random MOMDP.")
    parser.add_argument('-suc', type=int, default=4, help="The number of successors. Only used with the random MOMDP.")
    parser.add_argument('-noise', type=float, default=0.1, help="The stochasticity in state transitions.")
    parser.add_argument('-seed', type=int, default=1, help="The seed for random number generation. ")
    parser.add_argument('-num_iters', type=int, default=500, help="The number of iterations to run PQL for.")
    parser.add_argument('-max_t', type=int, default=1000, help="The maximum timesteps per episode.")
    parser.add_argument('-gamma', type=float, default=1, help="The discount factor for expected rewards.")
    parser.add_argument('-epsilon', type=float, default=0.5, help="How much error we tolerate on each objective.")
    parser.add_argument('-decimals', type=int, default=1, help="The number of decimals to include for each return.")
    parser.add_argument('-dir', type=str, default='results', help='The directory to save all results to.')
    parser.add_argument('-novec', type=int, default=10, help='The number of best vectors to keep.')

    args = parser.parse_args()

    env_name = args.env

    if env_name == 'RandomMOMDP-v0':
        env = gym.make('RandomMOMDP-v0', nstates=args.states, nobjectives=args.obj, nactions=args.act, nsuccessor=args.suc, seed=args.seed)
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        num_objectives = env._nobjectives
        num_successors = args.suc
        transition_function = env._transition_function
        reward_function = env._old_reward_function
        ref_point = np.zeros(num_objectives)
    elif env_name == 'RandomMOMDP-v1':
        env = gym.make('RandomMOMDP-v0', nstates=args.states, nobjectives=args.obj, nactions=args.act, nsuccessor=args.suc, seed=args.seed)
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        num_objectives = env._nobjectives
        num_successors = args.suc
        transition_function = env._transition_function
        reward_function = env._reward_function
        ref_point = np.zeros(num_objectives)
    else:
        env = gym.make('DeepSeaTreasure-v0', seed=args.seed, noise=args.noise)
        num_states = env.nS
        num_actions = env.nA
        num_objectives = 2
        num_successors = env.nS
        transition_function = env._transition_function
        reward_function = env._reward_function
        ref_point = np.array([0, -25])

    seed = args.seed
    gamma = args.gamma
    epsilon = args.epsilon
    decimals = args.decimals
    num_iters = args.num_iters
    max_t = args.max_t
    novec = args.novec
    np.random.seed(seed)
    Data = namedtuple('Data', ['vs', 'N', 's', 'a', 'ns'])

    path_data = args.dir
    mkdir_p(path_data)
    file = f'PQL_s{num_states}_a{num_actions}_o{num_objectives}_ss{args.suc}_seed{args.seed}_novec{novec}'

    pcs, dataset = run_pql(env, num_iters=num_iters, max_t=max_t, decimals=decimals, epsilon=epsilon, gamma=gamma, novec=novec)  # Run PQL.

    print_pcs(pcs)
    save_momdp(path_data, file, num_states, num_objectives, num_actions, num_successors, seed, transition_function,
               reward_function, epsilon, gamma)
    save_pcs(pcs, file, path_data, num_objectives)
    save_training_data(dataset, num_objectives, path_data, file)
