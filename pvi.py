import argparse
import time
import copy
import itertools
import gym

import pandas as pd
import numpy as np

from collections import namedtuple
from utils import mkdir_p, save_training_data, check_converged, print_pcs, save_pcs, save_momdp, get_best

from gym.envs.registration import register


register(
        id='RandomMOMDP-v0',
        entry_point='randommomdp:RandomMOMDP',
)

register(
        id='DeepSeaTreasure-v0',
        entry_point='deep_sea_treasure:DeepSeaTreasureEnv',
)


def pvi(decimals=4, epsilon=0.05, gamma=0.8, novec=10):
    """
    This function will run the Pareto Value Iteration algorithm.
    :param decimals: number of decimals to which the value vector should be rounded.
    :param epsilon: closeness to PCS.
    :param gamma: discount factor.
    :return: A set of non-dominated vectors per state in the MOMDP.
    """
    start = time.time()
    nd_vectors = [[{tuple(np.zeros(num_objectives))} for _ in range(num_actions)] for _ in range(num_states)]  # Q-set
    nd_vectors_update = copy.deepcopy(nd_vectors)
    #dataset = []
    run = 0  # For printing purposes.

    while run < 1000:  # We execute the algorithm until convergence.
        print(f'Value Iteration number: {run}')
        dataset = []

        for state in range(num_states):  # Loop over all states.
            print(f'Looping over state {state}')
            for action in range(num_actions):  # Loop over all actions possible in this state.
                candidate_vectors = set()  # A set of new candidate non-dominated vectors for this state action.
                next_states = np.where(transition_function[state, action, :] > 0)[0]  # Next states with prob > 0
                lv = []  # An empty list that will hold a list of vectors for each next state.

                for next_state in next_states:  # Loop over all states.
                    # We take the union of all state-action non dominated vectors.
                    # We then only keep the non dominated vectors.
                    # We cast the resulting set to a list for later processing.
                    lv.append(list(get_best(set().union(*nd_vectors[next_state]), max_points=novec)))

                # This cartesian product will contain tuples with a reward vector for each next state.
                cartesian_product = itertools.product(*lv)

                for next_vectors in cartesian_product:  # Loop over these tuples containing next vectors.
                    future_reward = np.zeros(num_objectives)  # The future reward associated with these next vectors.
                    N = np.zeros(num_objectives)  # The component of V from value vectors for the next state.

                    for idx, next_state in enumerate(next_states):
                        transition_prob = transition_function[state, action, next_state]  # The transition probability.
                        reward = reward_function[state, action, next_state]  # The reward associated with this.
                        next_vector = np.array(next_vectors[idx])  # The vector obtained in the next state.
                        disc_future_reward = gamma * next_vector  # The discounted future reward.
                        contribution = transition_prob * (reward + disc_future_reward)  # The contribution of this vector.
                        future_reward += contribution  # Add it to the future reward.
                        N += transition_prob * next_vector  # Add the component of V from next value vectors to N.

                    future_reward = tuple(np.around(future_reward, decimals=decimals))  # Round the future reward.
                    N = tuple(np.around(N, decimals=decimals))  # Round N.

                    for next_state, next_vector in zip(next_states, next_vectors):  # Add the trajectory to the dataset.
                        dataset.append(Data(next_vector, N, state, action, next_state))

                    candidate_vectors.add(future_reward)  # Add this future reward as a candidate.

                nd_vectors_update[state][action] = get_best(candidate_vectors, max_points=novec)  # Save ND for updating later.
        if check_converged(nd_vectors_update, nd_vectors, epsilon):  # Check if we converged already.
            #save_training_data(dataset, num_objectives, path_data, file)
            break  # If converged, break from the while loop and save data
        else:
            nd_vectors = copy.deepcopy(nd_vectors_update)  # Else perform a deep copy an go again.
            run += 1

    end = time.time()
    elapsed_seconds = (end - start)
    print("Seconds elapsed: " + str(elapsed_seconds))

    return nd_vectors_update, dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, default='RandomMOMDP-v0', help="The environment to run PVI on.")
    parser.add_argument('-states', type=int, default=10, help="The number of states. Only used with the random MOMDP.")
    parser.add_argument('-obj', type=int, default=2, help="The number of objectives. Only used with the random MOMDP.")
    parser.add_argument('-act', type=int, default=2, help="The number of actions. Only used with the random MOMDP.")
    parser.add_argument('-suc', type=int, default=4, help="The number of successors. Only used with the random MOMDP.")
    parser.add_argument('-noise', type=float, default=0.0, help="The stochasticity in state transitions.")
    parser.add_argument('-seed', type=int, default=42, help="The seed for random number generation. ")
    parser.add_argument('-gamma', type=float, default=0.8, help="The discount factor for expected rewards.")
    parser.add_argument('-epsilon', type=float, default=0.1, help="How much error we tolerate on each objective.")
    parser.add_argument('-decimals', type=int, default=2, help="The number of decimals to include for each return.")
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
    elif env_name == 'RandomMOMDP-v1':
        env = gym.make('RandomMOMDP-v0', nstates=args.states, nobjectives=args.obj, nactions=args.act, nsuccessor=args.suc, seed=args.seed)
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        num_objectives = env._nobjectives
        num_successors = args.suc
        transition_function = env._transition_function
        reward_function = env._reward_function
    else:
        env = gym.make('DeepSeaTreasure-v0', seed=args.seed, noise=args.noise)
        num_states = env.nS
        num_actions = env.nA
        num_objectives = 2
        num_successors = env.nS
        transition_function = env._transition_function
        reward_function = env._reward_function

    seed = args.seed
    gamma = args.gamma
    epsilon = args.epsilon
    decimals = args.decimals
    novec = args.novec
    np.random.seed(seed)
    Data = namedtuple('Data', ['vs', 'N', 's', 'a', 'ns'])

    path_data = args.dir
    mkdir_p(path_data)
    file = f'PVI_s{num_states}_a{num_actions}_o{num_objectives}_ss{args.suc}_seed{args.seed}_novec{novec}'

    pcs, NN_data = pvi(decimals=decimals, epsilon=epsilon, gamma=gamma, novec=novec)  # Run PVI.

    #print_pcs(pcs)
    save_training_data(NN_data)
    save_momdp(path_data, file, num_states, num_objectives, num_actions, num_successors, seed, transition_function, reward_function, epsilon, gamma)
    save_pcs(pcs, file, path_data, num_objectives)
