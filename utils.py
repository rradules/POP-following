import errno
import os
import json
import math

import numpy as np
import pandas as pd


def mkdir_p(path):
    """
    This function will create the necessary path if it does not exist yet.
    :param path:
    :return:
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_non_dominated(candidates):
    """
    This function returns the non-dominated subset of elements.
    :param candidates: The input set of candidate vectors.
    :return: The non-dominated subset of this input set.
    Source: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    candidates = np.array(list(candidates))
    # sort candidates by decreasing sum of coordinates
    candidates = candidates[candidates.sum(1).argsort()[::-1]]
    # initialize a boolean mask for undominated points
    # to avoid creating copies each iteration
    nd = np.ones(candidates.shape[0], dtype=bool)
    for i in range(candidates.shape[0]):
        # process each point in turn
        n = candidates.shape[0]
        if i >= n:
            break
        # find all points not dominated by i
        # since points are sorted by coordinate sum
        # i cannot dominate any points in 1,...,i-1
        nd[i + 1:n] = (candidates[i + 1:] > candidates[i]).any(1)
        # keep points non-dominated so far
        candidates = candidates[nd[:n]]

    non_dominated = set()
    for candidate in candidates:
        non_dominated.add(tuple(candidate))  # Add the non dominated vectors to a set again.
    return non_dominated


def is_dominated(candidate, vectors):
    """
    This function checks if a candidate vector is dominated by another vector in the set.
    :param candidate: The candidate vector.
    :param vectors: A set of vectors with type Tuple.
    :return: Boolean whether the candidate is dominated.
    """
    if not vectors:
        return False
    else:
        for vector in vectors:
            vector = np.array(vector)
            if not (candidate == vector).all():
                dominated = candidate <= vector
                if dominated.all():
                    return True
        return False


def print_pcs(pcs):
    """
    This function prints the Pareto coverage set.
    :param pcs: The pareto coverage set.
    :return: /
    """
    for state, sets in enumerate(pcs):
        for action, set in enumerate(sets):
            print(f'State {state} and action {action}: {repr(set)}')


def save_pcs(nd_vectors, file, path_data, num_objectives):
    """
    This function will save the generated pareto coverage set to a CSV file.
    :param nd_vectors: A set of non-dominated vectors per state.
    :param file: file name.
    :param path_data: directory path for saving the file.
    :param num_objectives: number of objectives.
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
    df.to_csv(f'{path_data}/PCS_{file}.csv', index=False)


def save_momdp(path, file, num_states, num_objectives, num_actions, num_successors, seed, transition_function,
               reward_function, epsilon, gamma):
    """
    This function saves all MOMDP info to a JSON file.
    :param path: The directory to save the new file in.
    :param file: The name of the info file.
    :param num_states: The number of states in the MOMDP.
    :param num_objectives: The number of objectives in the MOMDP.
    :param num_actions: The number of actions in the MOMDP.
    :param num_successors: The number of successor states in the MOMDP.
    :param seed: The random seed that was used.
    :param transition_function: The transition function in the MOMDP.
    :param reward_function: The reward function in the MOMDP.
    :param epsilon: The epsilon value for PVI.
    :param gamma: The gamma value for PVI.
    :return: /
    """
    info = {
        'states': num_states,
        'objectives': num_objectives,
        'actions': num_actions,
        'successors': num_successors,
        'seed': seed,
        'transition': transition_function.tolist(),
        'reward': reward_function.tolist(),
        'epsilon': epsilon,
        'gamma': gamma
    }
    json.dump(info, open(f'{path}/{file}.json', "w"))


def check_converged(new_nd_vectors, old_nd_vectors, epsilon):
    """
    This function checks if the PCS has converged.
    :param new_nd_vectors: The updated PCS.
    :param old_nd_vectors: The old PCS.
    :param epsilon: Distance to PCS.
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
