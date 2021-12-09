import random
import time
import gym
import torch
import json
import argparse

import pandas as pd
import numpy as np

from pop_nn import POP_NN, POP_NN_SDST
from utils import is_dominated, additive_epsilon_metric
from pop_ls import popf_local_search, popf_iter_local_search, toStavs
from gym.envs.registration import register

register(
    id='RandomMOMDP-v0',
    entry_point='randommomdp:RandomMOMDP',
)

register(
    id='DeepSeaTreasure-v0',
    entry_point='deep_sea_treasure:DeepSeaTreasureEnv',
)


def select_action(state, pcs, objective_columns, value_vector):
    """
    This function selects an action by looking in the local PCS for the closest value vector.
    :param state: The state to find an action for.
    :param pcs: The complete PCS.
    :param objective_columns: The column names for the objectives.
    :param value_vector: The value vector to find the closest match to.
    :return: The action associated with the closest match.
    """
    Q_next = pcs.loc[pcs['State'] == state]
    i_min = np.linalg.norm(Q_next[objective_columns] - value_vector, axis=1).argmin()
    action = Q_next['Action'].iloc[i_min]
    return action


def get_value(state, action, pcs, objective_columns, value_vector):
    """
    This function gets the closest value vector in a state and action.
    :param state: The state.
    :param action: The action.
    :param pcs: The complete PCS.
    :param objective_columns: The column names for the objectives.
    :param value_vector: The value vector to find the closest match to.
    :return: The closest value vector.
    """
    Q_next = pcs.loc[(pcs['State'] == state) & (pcs['Action'] == action)]
    i_min = np.linalg.norm(Q_next[objective_columns] - value_vector, axis=1).argmin()
    value = Q_next[objective_columns].iloc[i_min].values
    return value


def set_seeds(seed):
    """
    This function sets all seeds for random number generation. This ensures reproducability.
    :param seed: The requested seed.
    :return: /
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def load_model(model_filename, layers, activation):
    """
    This function loads a neural network and sets it to the correct mode.
    :param model_filename: The filename for the neural network parameters.
    :param layers: The number of nodes per layer as a list.
    :param activation: The type of activation function to use in the model.
    :return: The neural network.
    """
    d_in = num_objectives + 3  # Input dimensions for the neural network.
    d_out = num_objectives  # Output dimensions for the neural network.
    layers.insert(0, d_in)
    layers.append(d_out)

    if activation == 'ReLU':
        model = POP_NN(layers)
    else:
        model = POP_NN_SDST(layers)
    model.load_state_dict(torch.load(model_filename))
    model.eval()
    torch.no_grad()
    return model


def setup_experiment(env, pcs, objective_columns):
    """
    This function setups an experiment.
    :param env: The environment.
    :param pcs: The complete PCS.
    :param objective_columns: The objective columns in the dataset.
    :return: The start state, the action to take and the selected value vector.
    """
    state = env.reset()
    state_pcs = pcs[['Action', 'Objective 0', 'Objective 1']].loc[pcs['State'] == state]
    value_vectors = [state_pcs[objective_columns].to_numpy()]

    while True:
        random_row = state_pcs.sample(n=1)
        action = random_row['Action'].iloc[0]
        value_vector = random_row[objective_columns].iloc[0].values
        if not is_dominated(value_vector, value_vectors):
            break

    return state, action, value_vector


def rollout_search(env, optimiser, state, action, follow_vector, max_time=200):
    """
    This function executes one rollout of a local search optimiser in the environment.
    :param env: The environment.
    :param optimiser: The optimiser for finding the next action.
    :param state: The first state.
    :param action: The first action to take.
    :param follow_vector: The value vector to follow.
    :param max_time: The maximum of steps allowed to be taken in the environment.
    :return: The accumulated discounted returns and average optimisation scores.
    """
    time = 0
    done = False

    returns = np.zeros(num_objectives)
    cur_disc = 1
    total_opt_val = 0

    while time < max_time and not done:
        next_state, reward_vec, done, info = env.step(action)
        returns += cur_disc * reward_vec  # Accumulate discounted returns.
        cur_disc *= gamma  # Adjust the discounting.

        if follow_vector is not None:
            n_vector = (follow_vector - reward_vec) / gamma
            next_probs = transition_function[state][action]
            problem = toStavs(next_probs, pcs)
            cur_vector, score, action, follow_vector = optimiser(problem, n_vector, next_state)
            total_opt_val += score
        else:
            print(f'Warning: No value vector to follow is selected. This should not happen.')
            action = env.action_space.sample()

        state = next_state
        time += 1

    avg_opt_val = total_opt_val / time  # Record the average optimisation score in this rollout.
    return returns, avg_opt_val


def rollout_nn(env, model, state, action, follow_vector, max_time=200):
    """
    This function executes one rollout of the POPF neural network in the environment.
    :param env: The environment.
    :param model: The neural network.
    :param state: The first state.
    :param action: The first action.
    :param follow_vector: The value vector to follow.
    :param max_time: The maximum of steps allowed to be taken in the environment.
    :return: The accumulated discounted returns and average optimisation scores.
    """
    time = 0
    done = False

    returns = np.zeros(num_objectives)
    cur_disc = 1

    while time < max_time and not done:
        next_state, reward_vec, done, info = env.step(action)
        returns += cur_disc * reward_vec  # Accumulate discounted returns.
        cur_disc *= gamma  # Adjust the discounting.

        n_vector = (follow_vector - reward_vec) / gamma
        norm_n_vector = (n_vector - d_min) / (d_max - d_min)

        input_nn = [state / num_states, action / num_actions, next_state / num_states]  # Todo: states and actions should be encoded smarter.
        input_nn.extend(norm_n_vector)
        norm_follow_vector = model.forward(torch.tensor(input_nn, dtype=torch.float32))[0].detach().numpy()
        follow_vector = norm_follow_vector * (d_max - d_min) + d_min
        action = select_action(next_state, pcs, objective_columns, follow_vector)
        state = next_state

    return returns, 0


def run_experiment(env, optimiser, rollout, state, action, value_vector):
    """
    This function runs a complete experiment.
    :param env: The environment.
    :param optimiser: The optimiser to use.
    :param rollout: The function with which to complete a rollout.
    :param state: The starting state.
    :param action: The starting action.
    :param value_vector: The value vector to follow.
    :return: The logs for this experiment.
    """
    follow_vector = (value_vector - d_min) / (d_max - d_min)
    total_returns = np.zeros(num_objectives)
    episode_logs = []

    for episode in range(episodes):
        start = time.time()
        env.reset()
        env._state = state  # Always select the same initial state.
        returns, score = rollout(env, optimiser, state, action, follow_vector)
        end = time.time()
        elapsed_seconds = (end - start)
        total_returns = total_returns + returns
        episode_log = [episode, returns[0], returns[1], elapsed_seconds, score]
        episode_logs.append(episode_log)

    average_returns = total_returns / episodes
    difference, epsilon = additive_epsilon_metric(average_returns, value_vector)

    # Save results from this trial.
    logs_df = pd.DataFrame(episode_logs, columns=['Episode', 'Value0', 'Value1', 'Runtime', 'Score'])

    print(f'Achieved an average return of: {average_returns}')
    print(f'This results in a difference: {difference}, and epsilon metric: {epsilon}')
    print(f'----------------------------------------')

    return logs_df


def run_perturb_experiment(env, state, action, value_vector, reps=10):
    """
    This function runs the perturbation experiments.
    :param env: The environment.
    :param state: The starting state.
    :param action: The starting action.
    :param value_vector: The selected value vector.
    :param reps: The number of repetitions to use for ILS as a default.
    :return: The perturbation logs.
    """
    perturbations = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    perturb_logs = []

    for perturb in perturbations:
        print(f'Running ils with {perturb} perturbation probability')
        optimiser = lambda a, b, c: popf_iter_local_search(a, b, c, reps=reps, pertrub_p=perturb)
        logs = run_experiment(env, optimiser, rollout_search, state, action, value_vector)
        logs['Perturbation'] = perturb
        perturb_logs.append(logs)

    perturb_logs = pd.concat(perturb_logs)
    perturb_logs['Optimiser'] = 'ils'
    perturb_logs['Repetitions'] = reps

    return perturb_logs


def run_reps_experiment(env, state, action, value_vector, perturb=0.3):
    """
    This function runs the experiment for the number of repetitions.
    :param env: The environment.
    :param state: The starting state.
    :param action: The starting action.
    :param value_vector: The selected value vector.
    :param perturb: The default perturbation probability for ILS.
    :return: The logs for the repetition experiment.
    """
    repetitions = [5, 10, 15, 20, 25, 30, 35, 40]
    reps_logs = []

    for opt_str in ['ils', 'mls']:
        for reps in repetitions:
            print(f'Running {opt_str} with {reps} repetitions')

            if opt_str == 'ils':
                optimiser = lambda a, b, c: popf_iter_local_search(a, b, c, reps=reps, pertrub_p=perturb)
            else:
                optimiser = lambda a, b, c: popf_iter_local_search(a, b, c, reps=reps, pertrub_p=1)

            logs = run_experiment(env, optimiser, rollout_search, state, action, value_vector)
            logs['Repetitions'] = reps
            logs['Optimiser'] = opt_str
            reps_logs.append(logs)

    reps_logs = pd.concat(reps_logs)
    reps_logs['Perturbation'] = perturb

    return reps_logs


def run_regular_experiment(env, state, action, value_vector):
    """
    This function executes a regular experiment.
    :param env: The environment.
    :param state: The starting state.
    :param action: The starting action.
    :param value_vector: The selected value vector.
    :return: The logs for this experiment.
    """
    experiment_logs = []

    for opt_str in ['nn', 'ls']:
        print(f'Running experiments for {opt_str}')

        if opt_str == 'nn':
            optimiser = neural_network
            rollout = rollout_nn
        else:
            optimiser = popf_local_search
            rollout = rollout_search

        logs = run_experiment(env, optimiser, rollout, state, action, value_vector)
        logs['Optimiser'] = opt_str
        experiment_logs.append(logs)

    experiment_logs = pd.concat(experiment_logs)
    experiment_logs['Repetitions'] = -1
    experiment_logs['Perturbation'] = -1

    return experiment_logs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-trials', type=int, default=1, help='Number of trials to run.')
    parser.add_argument('-episodes', type=int, default=1, help='The number of episodes to run each trial.')
    parser.add_argument('-states', type=int, default=110, help="number of states")
    parser.add_argument('-obj', type=int, default=2, help="number of objectives")
    parser.add_argument('-act', type=int, default=4, help="number of actions")
    parser.add_argument('-suc', type=int, default=4, help="number of successors")
    parser.add_argument('-seed', type=int, default=42, help="seed")
    parser.add_argument('-exp_seed', type=int, default=1, help="experiment seed")
    parser.add_argument('-novec', type=int, default=10, help="No of vectors")
    parser.add_argument('-method', type=str, default='PVI', help="Method")
    parser.add_argument('-batch', type=int, default=32, help="batch size")
    parser.add_argument('-noise', type=float, default=0.1, help="The stochasticity in state transitions.")
    parser.add_argument('-layers', nargs='+', type=int, default=[16, 8, 4],help='Number of nodes per layer in the neural network.')

    args = parser.parse_args()

    # Reload environment.
    if args.states < 100:
        env = gym.make('RandomMOMDP-v0', nstates=args.states, nobjectives=args.obj, nactions=args.act,
                       nsuccessor=args.suc, seed=args.seed)
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        num_objectives = env._nobjectives
        num_successors = args.suc
    else:
        env = gym.make('DeepSeaTreasure-v0', seed=args.seed, noise=args.noise)
        num_states = env.nS
        num_actions = env.nA
        num_objectives = 2
        num_successors = env.nS

    # Extract arguments.
    trials = args.trials
    episodes = args.episodes
    novec = args.novec
    method = args.method
    batch = args.batch
    layers = args.layers
    exp_seed = args.exp_seed

    set_seeds(exp_seed)  # Set seed for random number generation.

    # Get the files.
    dir = f'results'
    filename = f's{args.states}_a{args.act}_o{args.obj}_ss{args.suc}_seed{args.seed}_novec{novec}'

    model_filename = f'{dir}/model_{batch}_{method}_{filename}.pth'
    neural_network = load_model(model_filename, layers, 'ReLU')

    with open(f'{dir}/MOMDP_{method}_{filename}.json', "r") as read_file:
        env_info = json.load(read_file)

    pcs = pd.read_csv(f'{dir}/PCS_{method}_{filename}.csv')

    # Set random MOMDPs to the original transition and reward functions.
    transition_function = env_info['transition']
    reward_function = env_info['reward']
    env._transition_function = np.array(transition_function)
    env._reward_function = np.array(reward_function)

    # Set parameters.
    if num_states > 100:
        gamma = 1  # Discount factor
        with open(f'{dir}/normNN_{method}_{filename}.json', "r") as read_file:
            norm_info = json.load(read_file)
        d_min = norm_info['min']
        d_max = norm_info['max']
    else:
        gamma = 0.8
        d_min = 0
        d_max = 1

    # Setup PCS correctly.
    objective_columns = ['Objective 0', 'Objective 1']
    pcs[objective_columns] = pcs[objective_columns].apply(pd.to_numeric)
    pcs[['Action', 'State']] = pcs[['Action', 'State']].astype('int32')

    # Setup results files.
    res_cols = ['Trial', 'Episode', 'Optimiser', 'Value0', 'Value1', 'Runtime', 'Score', 'Repetitions', 'Perturbation']
    results_file = f'{dir}/results_{method}_{filename}_exp{exp_seed}_{batch}.csv'
    exp_columns = ['Trial', 'Value0', 'Value1']
    exp_file = f'{dir}/experiment_{method}_{filename}_exp{exp_seed}_{batch}.csv'
    exp_data = []

    for trial in range(trials):
        print(f'Starting trial {trial}')
        state, action, value_vector = setup_experiment(env, pcs, objective_columns)
        print(f'From state {state} with action {action} selected value vector: {value_vector}')

        exp_data.append([trial, value_vector[0], value_vector[1]])
        logs_frames = []

        regular_logs = run_regular_experiment(env, state, action, value_vector)
        logs_frames.append(regular_logs)

        perturb_logs = run_perturb_experiment(env, state, action, value_vector)
        logs_frames.append(perturb_logs)

        reps_logs = run_reps_experiment(env, state, action, value_vector)
        logs_frames.append(reps_logs)

        # Save results from this trial.
        trial_logs = pd.concat(logs_frames)
        trial_logs['Trial'] = trial
        trial_logs = trial_logs[res_cols]  # Rearrange the column order.
        trial_logs.to_csv(results_file, mode='a', index=False)

        print(f'Finished trial {trial}')
        print(f'--------------------------------------------------------------------------------')
