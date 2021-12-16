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
    The code provided in all the stackoverflow answers is wrong. Important changes have been made in this function.
    """
    candidates = np.array(list(candidates))  # Turn the input set into a numpy array.
    candidates = candidates[candidates.sum(1).argsort()[::-1]]  # Sort candidates by decreasing sum of coordinates.
    for i in range(candidates.shape[0]):  # Process each point in turn.
        n = candidates.shape[0]  # Check current size of the candidates.
        if i >= n:  # If we've eliminated everything up until this size we stop.
            break
        nd = np.ones(candidates.shape[0], dtype=bool)  # Initialize a boolean mask for undominated points.
        # find all points not dominated by i
        # since points are sorted by coordinate sum
        # i cannot dominate any points in 1,...,i-1
        nd[i + 1:] = np.any(candidates[i + 1:] > candidates[i], axis=1)
        candidates = candidates[nd]  # Grab only the non-dominated vectors using the generated bitmask.

    non_dominated = set()
    for candidate in candidates:
        non_dominated.add(tuple(candidate))  # Add the non dominated vectors to a set again.
    return non_dominated


def crowding_distance_assignment(nd_array):
    """
    This function calculates the crowding distance for each point in the set.
    :param nd_array: The non-dominated set as an array.
    :return: The crowding distances.
    """
    size = nd_array.shape[0]
    num_objectives = nd_array.shape[1]
    crowding_distances = np.zeros(size)

    sorted_ind = np.argsort(nd_array, axis=0)  # The indexes of each column sorted.
    maxima = np.max(nd_array, axis=0)  # The maxima of each objective.
    minima = np.min(nd_array, axis=0)  # The minima of each objective.

    for obj in range(num_objectives):  # Loop over all objectives.
        crowding_distances[sorted_ind[0, obj]] = np.inf  # Always include the outer points.
        crowding_distances[sorted_ind[-1, obj]] = np.inf
        norm_factor = maxima[obj] - minima[obj]

        for i in range(1, size-1):  # Loop over all other points.
            distance = nd_array[sorted_ind[i+1, obj], obj] - nd_array[sorted_ind[i-1, obj], obj]
            crowding_distances[sorted_ind[i, obj]] += distance / norm_factor

    return crowding_distances


def get_best(candidates, max_points=10):
    """
    This function gets the best points from the candidate set.
    :param candidates: The set of candidates.
    :param max_points: The maximum number of points in the final set.
    :return: A non dominated set that is potentially further pruned using crowding distance.
    """
    non_dominated = get_non_dominated(candidates)  # Get the non dominated points.

    if max_points is None:  # If we want to keep everything return the non-dominated vectors already.
        return non_dominated

    points_to_remove = len(non_dominated) - max_points  # Calculate the number of points left to remove.

    if points_to_remove > 0:  # If we still need to discard points.
        nd_array = np.array(list(non_dominated))  # Transform the set to an array.
        crowding_distances = crowding_distance_assignment(nd_array)  # Calculate the crowding distances.
        max_ind = np.argsort(crowding_distances)[points_to_remove:]  # Get the indices of the best points.
        best_points = nd_array[max_ind]  # Select the best points using these indices.

        best_set = set()  # Place everything back into a set.
        for point in best_points:
            best_set.add(tuple(point))  # Add the non dominated vectors to a set again.
        return best_set
    else:
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


def save_pcs(path, nd_vectors, num_objectives):
    """
    This function will save the generated pareto coverage set to a CSV file.
    :param path: directory path for saving the file.
    :param nd_vectors: A set of non-dominated vectors per state.
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
    df.to_csv(f'{path}/pcs.csv', index=False)


def save_momdp(path, num_states, num_objectives, num_actions, num_successors, seed, transition_function,
               reward_function, epsilon, gamma):
    """
    This function saves all MOMDP info to a JSON file.
    :param path: The directory to save the new file in.
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

    with open(f'{path}/momdp.json', 'w') as f:
        json.dump(info, f)


def save_experiment(path, min_pcs, max_pcs, mean_pcs, start_state_pcs, values, seed):
    """
    This function saves an experiments metadata.
    :param path: The path to save the results to.
    :param min_pcs: The minimum size of all local PCSs.
    :param max_pcs: The maximum size of all local PCSs.
    :param mean_pcs: The mean size of all local PCSs.
    :param start_state_pcs: The size of the PCS in the start state.
    :param values: The selected value vectors in the experiment.
    :param seed: The seed for the experiment.
    :return: /
    """
    exp_data = {'min': float(min_pcs),
                'max': float(max_pcs),
                'mean': float(mean_pcs),
                's0': float(start_state_pcs),
                'values': values,
                'seed': seed}

    with open(path, 'w') as f:
        json.dump(exp_data, f)


def save_training_data(path, dataset, num_objectives):
    """
    This function saves the dataset in a structured way for later use in training a neural network.
    :param path: The path to the directory for saving the data.
    :param dataset: The created dataset from PVI.
    :param num_objectives: The number of objectives.
    :return: /
    """
    columns = ['s', 'a', 'ns']
    columns.extend([f'N{i}' for i in range(num_objectives)])
    columns.extend([f'vs{i}' for i in range(num_objectives)])

    data = []

    for instance in dataset:
        s = [instance.s]
        a = [instance.a]
        ns = [instance.ns]
        N = list(instance.N)
        vs = list(instance.vs)
        data.append(s + a + ns + N + vs)

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(f'{path}/training_data.csv', index=False)


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


def additive_epsilon_metric(new_vec, pareto_vec):
    """
    This function returns the additive epsilon metric between a new vector and a vector on the known Pareto front.
    :param new_vec: The new vector.
    :param pareto_vec: The vector on the Pareto front.
    :return: The difference and additive epsilon metric between the two vectors. Bounded by the interval [0, inf).
    """
    difference = pareto_vec - new_vec
    max_diff = np.max(difference)
    epsilon = max(0, max_diff)
    return difference, epsilon


def multiplicative_epsilon_metric(new_vec, pareto_vec):
    """
    This function returns the multiplicative epsilon metric between a new vector and a vector on the known Pareto front.
    :param new_vec: The new vector.
    :param pareto_vec: The vector on the Pareto front.
    :return: The multiplicative epsilon metric between the two vectors. Bounded by the interval [0, inf).
    Note that this does not go well with negative values.
    """
    percentage_change = (pareto_vec - new_vec) / np.abs(new_vec)
    max_change = np.max(percentage_change)
    epsilon = max(0, max_change)
    return epsilon
