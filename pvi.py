import argparse
import time
import copy
import itertools
import gym

import numpy as np

from collections import namedtuple
from utils import mkdir_p, save_training_data, check_converged, print_pcs, save_pcs, save_momdp, get_best, load_pcs
from pop_nn import train_batch, load_network
from replay_buffer import ReplayBuffer

from gym.envs.registration import register


register(
        id='RandomMOMDP-v0',
        entry_point='envs.randommomdp:RandomMOMDP',
)

register(
        id='DeepSeaTreasure-v0',
        entry_point='envs.deep_sea_treasure:DeepSeaTreasureEnv',
)


def pvi(init_pcs, max_iter=1000, decimals=4, epsilon=0.05, gamma=0.8, max_vec=10, save_every=10):
    """
    This function will run the Pareto Value Iteration algorithm.
    :param init_pcs: The initial PCS to use as a starting point for PVI.
    :param max_iter: The maximum number of iterations to run PVI for.
    :param decimals: number of decimals to which the value vector should be rounded.
    :param epsilon: closeness to PCS.
    :param gamma: discount factor.
    :param save_every: Save the current PCS and neural network dataset every number of iterations.
    :return: A set of non-dominated vectors per state in the MOMDP.
    """
    if train:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = load_network(model_str, num_objectives, dropout).to(device)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        replay_buffer = ReplayBuffer(capacity=50000, batch_size=64)

    start = time.time()
    dataset = []
    nd_vectors = init_pcs
    nd_vectors_update = copy.deepcopy(nd_vectors)

    last_iter = max_iter - 2  # Denotes the start of the last run.
    is_last = False
    save_iteration = False

    for run in range(max_iter):  # We execute the algorithm for a number of iterations.
        print(f'Value Iteration number: {run}')

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
                    lv.append(list(get_best(set().union(*nd_vectors[next_state]), max_points=max_vec)))

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

                    if save_iteration or is_last or train:
                        for next_state, next_vector in zip(next_states, next_vectors):  # Add the trajectory to the dataset.
                            if save_iteration or is_last:
                                dataset.append(Data(next_vector, N, state, action, next_state))
                            if train:
                                replay_buffer.append(next_vector, N, state, action, next_state)

                    if train and replay_buffer.can_sample():
                        batch = replay_buffer.sample()
                        data = torch.tensor(batch[:, :-num_objectives], dtype=torch.float32)
                        target = torch.tensor(batch[:, -num_objectives], dtype=torch.float32).unsqueeze(dim=1)
                        model, optimizer, loss = train_batch(model, loss_function, optimizer, data, target)
                        print(f'Loss in run {run}: {loss}')

                    candidate_vectors.add(future_reward)  # Add this future reward as a candidate.

                nd_vectors_update[state][action] = get_best(candidate_vectors, max_points=max_vec)  # Save ND for updating later.
        if is_last:
            break
        if save_iteration:
            print("Saving this round")
            save_training_data(data_dir, dataset, num_objectives)
            save_pcs(pcs_dir, nd_vectors_update, num_objectives)
            save_iteration = False
            dataset = []
        if run % save_every == 0:
            save_iteration = True
        if check_converged(nd_vectors_update, nd_vectors, epsilon) or run >= last_iter:  # Check if we converged or are in the last run.
            max_vec = None  # Save everything in the last iteration.
            is_last = True
        nd_vectors = copy.deepcopy(nd_vectors_update)  # Else perform a deep copy an go again.

    end = time.time()
    elapsed_seconds = (end - start)
    print("Seconds elapsed: " + str(elapsed_seconds))

    return nd_vectors_update, dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-train', type=bool, default=False, help='Train a neural network online')
    parser.add_argument('-model', type=str, default='MlpSmall', help="The network architecture to use.")
    parser.add_argument('-dropout', type=float, default=0., help='Dropout rate for the neural network')
    parser.add_argument('-cont', type=bool, default=True, help='Whether or not to continue from a start PCS.')
    parser.add_argument('-dir', type=str, default='results/PVI/SDST', help="The directory to save the results.")
    parser.add_argument('-env', type=str, default='SDST', help="The environment to run PVI on.")
    parser.add_argument('-states', type=int, default=10, help="The number of states. Only used with the random MOMDP.")
    parser.add_argument('-obj', type=int, default=2, help="The number of objectives. Only used with the random MOMDP.")
    parser.add_argument('-act', type=int, default=2, help="The number of actions. Only used with the random MOMDP.")
    parser.add_argument('-suc', type=int, default=4, help="The number of successors. Only used with the random MOMDP.")
    parser.add_argument('-noise', type=float, default=0.1, help="The stochasticity in state transitions.")
    parser.add_argument('-seed', type=int, default=42, help="The seed for random number generation. ")
    parser.add_argument('-num_iters', type=int, default=200, help="The maximum number of iterations to run PVI for.")
    parser.add_argument('-gamma', type=float, default=1, help="The discount factor for expected rewards.")
    parser.add_argument('-epsilon', type=float, default=0.1, help="How much error we tolerate on each objective.")
    parser.add_argument('-decimals', type=int, default=2, help="The number of decimals to include for each return.")
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
    num_iters = args.num_iters
    np.random.seed(seed)
    Data = namedtuple('Data', ['vs', 'N', 's', 'a', 'ns'])
    cont = args.cont
    train = args.train
    model_str = args.model
    dropout = args.dropout

    res_dir = args.dir
    pcs_dir = f'{res_dir}/PCS'
    data_dir = f'{res_dir}/data'

    mkdir_p(pcs_dir)
    mkdir_p(data_dir)

    if train:  # Only import these modules when we actually want to train.
        import torch
        import torch.nn as nn

    init_pcs = load_pcs(cont, pcs_dir, num_states, num_actions, num_objectives)
    pcs, dataset = pvi(init_pcs, max_iter=num_iters, decimals=decimals, epsilon=epsilon, gamma=gamma, max_vec=novec)  # Run PVI.

    print_pcs(pcs)
    save_training_data(data_dir, dataset, num_objectives)
    save_momdp(res_dir, num_states, num_objectives, num_actions, num_successors, seed, transition_function, reward_function, epsilon, gamma)
    save_pcs(pcs_dir, pcs, num_objectives)
