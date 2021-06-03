import argparse
import json
import time
import copy
import math
import gym

import pandas as pd
import numpy as np

from utils import mkdir_p, get_non_dominated

from gym.envs.registration import register


def check_converged(new_nd_vectors, old_nd_vectors):
    """
    This function checks if the PCS has converged.
    :param new_nd_vectors: The updated PCS.
    :param old_nd_vectors: The old PCS.
    :return: Boolean whether the PCS has converged.
    """
    for state in range(len(new_nd_vectors)):
        for new_set, old_set in zip(new_nd_vectors[state], old_nd_vectors[state]):
            min_epsilon = math.inf  # Initial value is positive infinity because we need the minimum.
            for new_vec in new_set:
                for old_vec in old_set:
                    diff = np.asarray(new_vec) - np.asarray(old_vec)
                    min_epsilon = min(min_epsilon, np.max(diff))  # We need the max difference.
                if min_epsilon > epsilon:  # Early stop if the minimum epsilon for a vector is already above the threshold.
                    return False
    return True


def pvi():
    """
    This function will run the Pareto Value Iteration algorithm.
    :return: A set of non-dominated vectors per state in the MOMDP.
    """
    start = time.time()
    nd_vectors = [[{tuple(np.zeros(num_objectives))} for _ in range(num_actions)] for _ in range(num_states)]  # Q-set
    nd_vectors_update = copy.deepcopy(nd_vectors)
    nn_dataset = {}

    run = 0  # For printing purposes.

    while True:  # We execute the algorithm until convergence.
        print(f'Value Iteration number: {run}')
        for state in range(num_states):  # Loop over all states.
            dict_v = {}
            dict_c = {}
            dict_cand = {}
            dict_future = {}

            for action in range(num_actions):  # Loop over all actions possible in this state.
                candidate_vectors = set()  # A set of new candidate non-dominated vectors for this state action.
                reward = reward_function[state, action]  # Get the reward from taking the action in the state.
                future_rewards = {tuple(np.zeros(num_objectives))}  # The vectors that can be obtained in next states.
                next_states = np.where(transition_function[state, action, :] > 0)[0]  # Next states with prob > 0

                for next_state in next_states:  # Loop over the next states
                    transition_prob = transition_function[state, action, next_state]  # Probability of this transition.
                    new_future_rewards = set()  # Empty set that will hold the updated future rewards
                    for next_action in range(num_actions):
                        next_state_action_nd_vectors = nd_vectors[next_state][next_action]  # Non dominated vectors from the next state.

                        for curr_vec in future_rewards:  # Current set of future rewards.
                            for nd_vec in next_state_action_nd_vectors:  # Loop over the non-dominated vectors inn this next state.
                                future_reward = np.array(curr_vec) + transition_prob * np.array(nd_vec)
                                future_reward = tuple(np.around(future_reward, decimals=decimals))
                                new_future_rewards.add(future_reward)
                                dict_future[future_reward] = [nd_vec, state, action, next_state]

                    future_rewards = get_non_dominated(new_future_rewards)  # Update the future rewards with the updated set.
                    dict_future_update = {tuple(rew): dict_future[tuple(rew)] for rew in future_rewards}
                    dict_v.update(dict_future_update)
                    assert (len(future_rewards) == len(dict_future_update))

                for future_reward in future_rewards:
                    value_vector = tuple(reward + gamma * np.array(future_reward)) # Calculate estimate of the value vector.
                    value_vector = tuple(np.around(value_vector, decimals=decimals))
                    candidate_vectors.add(value_vector)

                    dict_cand[value_vector] = [future_reward, reward]

                nd_vectors_update[state][action] = candidate_vectors  # Update the non-dominated set.
                dict_cand_update = {tuple(val): dict_cand[tuple(val)] for val in nd_vectors_update[state][action]}
                dict_c.update(dict_cand_update)
                assert(len(nd_vectors_update[state][action]) == len(dict_cand_update))
                # here we can filter dict_v again if it gets too slow
                nn_dataset[state] = [dict_v, dict_c]

        if check_converged(nd_vectors_update, nd_vectors):
            columns = ['s', 'a', 'ns']
            columns.extend([f'N{i}' for i in range(num_objectives)])
            columns.extend([f'vs{i}' for i in range(num_objectives)])

            print(columns)

            data = []
            for state in nn_dataset:
                dict_v = nn_dataset[state][0]
                dict_c = nn_dataset[state][1]
                for value in dict_c:
                    future_value, reward = dict_c[value]
                    # N = (val - rew)/gamma
                    N = (value - reward)/gamma
                    # nd next vectors, s, a, ns
                    info = dict_v[tuple(future_value)]
                    entry = [info[1], info[2], info[3]]
                    entry.extend(N)
                    entry.extend(info[0])
                    data.append(entry)
                    print(data)
                assert(state == info[1])
            df = pd.DataFrame(data, columns=columns)
            df.to_csv(f'{path_data}NN_{file}.csv', index=False)
            break  # If converged, break from the while loop and save data
        else:
            nd_vectors = copy.deepcopy(nd_vectors_update)  # Else perform a deep copy an go again.
            run += 1

    end = time.time()
    elapsed_seconds = (end - start)
    print("Seconds elapsed: " + str(elapsed_seconds))

    return nd_vectors_update


def save_vectors(nd_vectors, file):
    """
    This function will save the generated pareto coverage set to a CSV file.
    :param nd_vectors: A set of non-dominated vectors per state.
    :return: /
    """

    columns = [f'Objective {i}' for i in range(num_objectives)]
    columns = ['State', 'Action'] + columns
    results = []
    for state, actions in enumerate(nd_vectors):
        for action, vectors in enumerate(actions):
            for vector in vectors:
                row = [state, action] + list(vector)
                results.append(row)
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(f'{path_data}PCS_{file}.csv', index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-states', type=int, default=10, help="number of states")
    parser.add_argument('-obj', type=int, default=2, help="number of objectives")
    parser.add_argument('-act', type=int, default=2, help="number of actions")
    parser.add_argument('-suc', type=int, default=4, help="number of successors")
    parser.add_argument('-seed', type=int, default=1, help="seed")

    args = parser.parse_args()

    register(
        id='RandomMOMDP-v0',
        entry_point='randommomdp:RandomMOMDP',
        reward_threshold=0.0,
        kwargs={'nstates': args.states, 'nobjectives': args.obj,
                'nactions': args.act, 'nsuccessor': args.suc, 'seed': args.seed}
    )

    env = gym.make('RandomMOMDP-v0')
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    num_objectives = env._nobjectives

    transition_function = env._transition_function
    reward_function = env._reward_function

    gamma = 0.8  # Discount factor
    epsilon = 0.05  # How close we want to go to the PCS.
    decimals = 4

    path_data = f'results/'  # /{mooc}/{hp.use_baseline}'
    mkdir_p(path_data)

    file = f'MPD_s{num_states}_a{num_actions}_o{num_objectives}_ss{args.suc}_seed{args.seed}'

    env_info = env.info
    env_info['epsilon'] = epsilon
    env_info['gamma'] = gamma

    nd_vectors = pvi()

    for idx, vectors in enumerate(nd_vectors):
        print(repr(idx), repr(vectors))

    save_vectors(nd_vectors, file)
    json.dump(env_info, open(f'{path_data}{file}.json', "w"))

