import gym
import pandas as pd
import numpy as np

from gym.envs.registration import register

register(
    id='RandomMOMDP-v0',
    entry_point='randommomdp:RandomMOMDP',
    reward_threshold=0.0,
    kwargs={'nstates': 4, 'nobjectives': 2, 'nactions': 2, 'nsuccessor': 2, 'seed': 1}
)

env = gym.make('RandomMOMDP-v0')
num_states = env.observation_space.n
num_actions = env.action_space.n
num_objectives = env._nobjectives

transition_function = env._transition_function
reward_function = env._reward_function

gamma = 0.9  # Discount factor


def check_dominated(candidate, vectors):
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
            dominated = candidate < np.array(vector)
            if dominated.all():
                return True
        return False


def pvi():
    """
    This function will run the Pareto Value Iteration algorithm.
    :return: A set of non-dominated vectors per state in the MOMDP.
    """
    nd_vectors = [{tuple(np.zeros(num_objectives))} for _ in range(num_states)]  # Set non-dominated vectors to zero.

    for run in range(10):  # We execute the algorithm a set amount of rounds.
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

                    future_rewards = new_future_rewards  # Update the future rewards with the updated set.
                for future_reward in future_rewards:
                    value_vector = reward + gamma * np.array(future_reward)  # Calculate estimate of the value vector.
                    candidate_vectors.add(tuple(value_vector))

            for vec in candidate_vectors:  # Loop over all the candidate vectors.
                dominated = check_dominated(np.array(vec), candidate_vectors)  # Check if the new vector is dominated.
                if not dominated:
                    new_nd_vectors.add(vec)  # If it is not, add it to the non-dominated set.

            nd_vectors[state] = new_nd_vectors  # Update the non-dominated set.

    return nd_vectors


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
