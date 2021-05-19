import copy
import math
import gym
import pandas as pd
import numpy as np

from gym.envs.registration import register

register(
    id='RandomMOMDP-v0',
    entry_point='randommomdp:RandomMOMDP',
    reward_threshold=0.0,
    kwargs={'nstates': 5, 'nobjectives': 2, 'nactions': 2, 'nsuccessor': 3, 'seed': 1}
)

env = gym.make('RandomMOMDP-v0')
num_states = env.observation_space.n
num_actions = env.action_space.n
num_objectives = env._nobjectives

transition_function = env._transition_function
reward_function = env._reward_function

gamma = 0.8  # Discount factor
epsilon = 0.1 # How close we want to go to the PCS.


def get_non_dominated(candidates):
    """returns the non-dominated subset of elements"""
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
        nd[i+1:n] = (candidates[i+1:] > candidates[i]).any(1)
        # keep points non-dominated so far
        candidates = candidates[nd[:n]]
    return candidates


def check_converged(new_nd_vectors, old_nd_vectors):
    """
    This function checks if the PCS has converged.
    :param new_nd_vectors: The updated PCS.
    :param old_nd_vectors: The old PCS.
    :return: Boolean whether the PCS has converged.
    """
    for new_set, old_set in zip(new_nd_vectors, old_nd_vectors):
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
    nd_vectors = [{tuple(np.zeros(num_objectives))} for _ in range(num_states)]  # Set non-dominated vectors to zero.
    nd_vectors_update = copy.deepcopy(nd_vectors)

    run = 0  # For printing purposes.

    while True:  # We execute the algorithm a set amount of rounds.
        print(f'Value Iteration number: {run}')
        for state in range(num_states):  # Loop over all states.
            candidate_vectors = set()  # A set of new candidate non-dominated vectors for this state.
            new_nd_vectors = set()  # Initialise an empty set to hold the new non-dominated vectors for this state.

            for action in range(num_actions):  # Loop over all actions possible in this state.
                reward = reward_function[state, action]  # Get the reward from taking the action in the state.
                future_rewards = {tuple(np.zeros(num_objectives))}  # The vectors that can be obtained in next states.
                next_states = np.where(transition_function[state, action, :] > 0)[0]  # Next states with prob > 0

                for next_state in next_states:  # Loop over the next states
                    next_state_nd_vectors = nd_vectors[next_state]  # Non dominated vectors from the next state.
                    transition_prob = transition_function[state, action, next_state]  # Probability of this transition.
                    new_future_rewards = set()  # Empty set that will hold the updated future rewards

                    for curr_vec in future_rewards:  # Current set of future rewards.
                        for nd_vec in next_state_nd_vectors:  # Loop over the non-dominated vectors inn this next state.
                            future_reward = np.array(curr_vec) + transition_prob * np.array(nd_vec)
                            new_future_rewards.add(tuple(future_reward))

                    future_rewards = new_future_rewards # Update the future rewards with the updated set.
                for future_reward in future_rewards:
                    value_vector = reward + gamma * np.array(future_reward)  # Calculate estimate of the value vector.
                    candidate_vectors.add(tuple(value_vector))
            print(f'Candidates: {len(candidate_vectors)} for state {state}')
            nd_vectors_update[state] = get_non_dominated(candidate_vectors)  # Update the non-dominated set.
            print(f'Nd values: {len(nd_vectors_update[state])} for state {state}')

        if check_converged(nd_vectors_update, nd_vectors):
            return nd_vectors_update  # If converged, return the latest non-dominated vectors.
        else:
            nd_vectors = copy.deepcopy(nd_vectors_update)  # Else perform a deep copy an go again.
            run += 1


def save_vectors(nd_vectors):
    """
    This function will save the generated pareto coverage set to a CSV file.
    :param nd_vectors: A set of non-dominated vectors per state.
    :return: /
    """
    columns = [f'Objective {i}' for i in range(num_objectives)]
    columns.insert(0, 'State')
    results = []
    for state, vectors in enumerate(nd_vectors):
        for vector in vectors:
            row = list(vector)
            row.insert(0, state)
            results.append(row)
    df = pd.DataFrame(results, columns=columns)
    df.to_csv('results.csv', index=False)


if __name__ == '__main__':
    nd_vectors = pvi()
    for idx, vectors in enumerate(nd_vectors):
        print(repr(idx), repr(vectors))
    save_vectors(nd_vectors)