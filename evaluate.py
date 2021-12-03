import gym
import torch
from gym.envs.registration import register
import json
import argparse
import pandas as pd
from pop_nn import POP_NN
from utils import is_dominated
import numpy as np
import random
import time
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
    Q_next = pcs.loc[pcs['State'] == state]
    i_min = np.linalg.norm(Q_next[objective_columns] - value_vector, axis=1).argmin()
    action = Q_next['Action'].iloc[i_min]
    #print(f'VV{value_vector}, {np.linalg.norm(Q_next[objective_columns] - value_vector, axis=1).min()}')
    return action


def get_value(state, action, pcs, objective_columns, value_vector):
    Q_next = pcs.loc[(pcs['State'] == state) & (pcs['Action'] == action)]
    i_min = np.linalg.norm(Q_next[objective_columns] - value_vector, axis=1).argmin()
    value = Q_next[objective_columns].iloc[i_min].values
    #print(value_vector, value)
    return value


def rollout(env, state0, action0, value_vector, pcs, gamma, max_time=200, optimiser=popf_local_search):
    # Assuming the state in the environment is indeed state0;
    # the reset needs to happen outside of this function
    time = 0
    stop = False
    action = action0
    state = state0
    returns = np.zeros(num_objectives)
    cur_disc = 1
    while time < max_time and not stop:
        if value_vector is None:
            action = env.action_space.sample()
        # action picked, now let's execute it
        next_state, reward_vec, done, info = env.step(action)
        returns += cur_disc * reward_vec
        # lowering the next timesteps forefactor:
        cur_disc *= gamma

        if value_vector is not None:
            n_vector = value_vector - reward_vec
            n_vector /= gamma
            next_probs = transition_function[state][action]
            problem = toStavs(next_probs, pcs)
            nm1, nm2, action, value_vector = optimiser(problem, n_vector, next_state)
            # print('.',end='', flush=True)

        state = next_state
        stop = done
        time += 1
    return returns


def eval_POP_NN(env, s_prev, a_prev, v_prev):
    # Load the NN model
    model = POP_NN(nnl)
    model.load_state_dict(torch.load(f'{path_data}ND_model_{batch}_{method}_{file}.pth'))
    model.eval()
    ret_vector = np.zeros(num_objectives)
    cur_disc = 1

    done = False
    with torch.no_grad():
        while not done:
            s_next, r_next, done, _ = env.step(a_prev)
            ret_vector += cur_disc * r_next
            cur_disc *= gamma
            # print(s_prev, a_prev, s_next, r_next, done)
            N = (v_prev - r_next) / gamma
            if method == 'PVI':
                N = np.around(N, decimals=2)
            inputNN = [s_prev / num_states, a_prev / num_actions, s_next / num_states]
            inputNN.extend(N)
            #print(f'NNinput {inputNN}')
            v_next = model.forward(torch.tensor(inputNN, dtype=torch.float32))[0].numpy()
            v_prev = v_next
            a_prev = select_action(s_next, pcs, objective_columns, v_next)
            s_prev = s_next

    # print(v0, ret_vector)
    return ret_vector


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-states', type=int, default=10, help="number of states")
    parser.add_argument('-obj', type=int, default=2, help="number of objectives")
    parser.add_argument('-act', type=int, default=2, help="number of actions")
    parser.add_argument('-suc', type=int, default=4, help="number of successors")
    parser.add_argument('-seed', type=int, default=42, help="seed")
    parser.add_argument('-exp_seed', type=int, default=1, help="experiment seed")
    parser.add_argument('-optimiser', type=str, default='nn', help="Optimiser")
    parser.add_argument('-reps', type=int, default=10, help="Reps")
    parser.add_argument('-novec', type=int, default=30, help="No of vectors")
    parser.add_argument('-method', type=str, default='PQL', help="Method")
    parser.add_argument('-batch', type=int, default=8, help="batch size")
    parser.add_argument('-noise', type=float, default=0.1, help="The stochasticity in state transitions.")
    parser.add_argument('-nnl', help='NN layer structure', type=lambda s: [int(item) for item in s.split(',')])

    args = parser.parse_args()

    # reload environment
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

    np.random.seed(args.exp_seed)
    random.seed(args.exp_seed)
    torch.manual_seed(args.exp_seed)

    num_states = env.observation_space.n
    num_actions = env.action_space.n
    num_objectives = args.obj
    novec = args.novec
    method = args.method
    batch = args.batch
    nnl = args.nnl

    # input output size
    d_in = num_objectives + 3
    d_out = num_objectives
    # init NN
    nnl.insert(0, d_in)
    nnl.append(d_out)

    path_data = f'results/'
    file = f's{args.states}_a{args.act}_o{args.obj}_ss{args.suc}_seed{args.seed}_novec{novec}'

    num_states = args.states
    num_actions = args.act
    num_objectives = args.obj

    objective_columns = ['Objective 0', 'Objective 1']

    if num_states < 100:
        gamma = 0.8  # Discount factor
    else:
        gamma = 1

    with open(f'{path_data}MOMDP_{method}_{file}.json', "r") as read_file:
        env_info = json.load(read_file)

    transition_function = env_info['transition']
    env._transition_function = np.array(transition_function)

    reward_function = env_info['reward']
    env._reward_function = np.array(reward_function)

    pcs = pd.read_csv(f'{path_data}PCS_{method}_{file}.csv')

    pcs[objective_columns] = pcs[objective_columns].apply(pd.to_numeric)
    pcs[['Action', 'State']] = pcs[['Action', 'State']].astype('int32')

    s0 = env.reset()
    dom = True
    subset = pcs[['Action', 'Objective 0', 'Objective 1']].loc[pcs['State'] == s0]
    cand = [subset[objective_columns].to_numpy()]

    # Select initial non-dominated value
    while dom:
        select = subset.sample()
        a0 = select['Action'].iloc[0]
        v0 = select[objective_columns].iloc[0].values
        dom = is_dominated(v0, cand)

    print(s0, a0, v0)
    times = 200

    opt_str = args.optimiser
    # 'ls', 'mls', 'ils', 'nn'
    # opt_str = 'nn'
    results = []
    lsrep = args.reps
    #lsrep = 10
    perturbations = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    lsreps = [5, 10, 15, 20, 25, 30, 35, 40]
    #  ['nn', 'ls', 'mls', 'ils']
    for opt_str in ['nn', 'ls', 'ils']:
        if opt_str == 'nn':
            print(f'Running {opt_str}.')
            acc = np.array([0.0, 0.0])
            for x in range(times):
                start = time.time()
                env.reset()
                env._state = s0
                returns = eval_POP_NN(env, s0, a0, v0)
                # print(f'{x+1}: {returns}', flush=True)
                end = time.time()
                elapsed_seconds = (end - start)
                acc = acc + returns
                results.append(np.append(returns, [x, elapsed_seconds, opt_str, -1, -1]))

            av = acc / times
            diff = v0 - av
            l = max(0, max(diff))
            print(f'{opt_str}: {l}, {diff}, vec={av}')

        elif opt_str == 'ls':
            optimiser = popf_local_search
            acc = np.array([0.0, 0.0])
            for x in range(times):
                start = time.time()
                env.reset()
                env._state = s0
                returns = rollout(env, s0, a0, v0, pcs, gamma, optimiser=optimiser)
                # print(f'{x+1}: {returns}', flush=True)
                end = time.time()
                elapsed_seconds = (end - start)
                acc = acc + returns
                # results.append(np.append(returns, [x, elapsed_seconds, opt_str]))
                results.append(np.append(returns, [x, elapsed_seconds, opt_str, -1, -1]))
            av = acc / times
            diff = v0 - av
            l = max(0, max(diff))
            print(f'{opt_str}: {l}, {diff}, vec={av}')

        elif opt_str == 'ils':
            for perturb in perturbations:
                print(f'Running {opt_str} with {perturb} perturbations.')
                func = lambda a, b, c: popf_iter_local_search(a, b, c, reps=lsrep, pertrub_p=perturb)
                optimiser = func
                acc = np.array([0.0, 0.0])
                for x in range(times):
                    start = time.time()
                    env.reset()
                    env._state = s0
                    returns = rollout(env, s0, a0, v0, pcs, gamma, optimiser=optimiser)
                    # print(f'{x+1}: {returns}', flush=True)
                    end = time.time()
                    elapsed_seconds = (end - start)
                    acc = acc + returns
                    # results.append(np.append(returns, [x, elapsed_seconds, opt_str]))
                    results.append(np.append(returns, [x, elapsed_seconds, opt_str, lsrep, perturb]))
                av = acc / times
                diff = v0 - av
                l = max(0, max(diff))
                print(f'{opt_str}: {l}, {diff}, vec={av}')

    for opt_str in ['mls', 'ils']:
        for lsrep in lsreps:
            print(f'Running {opt_str} with {lsreps} repetitions.')
            if opt_str == 'mls':
                perturb = 1
                func = lambda a, b, c: popf_iter_local_search(a, b, c, reps=lsrep, pertrub_p=perturb)
                optimiser = func
            elif opt_str == 'ils':
                perturb = 0.3
                func = lambda a, b, c: popf_iter_local_search(a, b, c, reps=lsrep, pertrub_p=perturb)
                optimiser = func

            acc = np.array([0.0, 0.0])
            for x in range(times):
                start = time.time()
                env.reset()
                env._state = s0
                returns = rollout(env, s0, a0, v0, pcs, gamma, optimiser=optimiser)
                # print(f'{x+1}: {returns}', flush=True)
                end = time.time()
                elapsed_seconds = (end - start)
                acc = acc + returns
                results.append(np.append(returns, [x, elapsed_seconds, opt_str, lsrep, perturb]))
            av = acc / times
            diff = v0 - av
            l = max(0, max(diff))
            print(f'{opt_str}, {lsrep}, {perturb}: {l}, {diff}, vec={av}')

    final_result = {'method': 'all', 'v0': v0.tolist()}
    json.dump(final_result,
              open(f'{path_data}ND_results_all_{method}_{file}_exp{args.exp_seed}_{batch}_reps{args.reps}.json', "w"))

    #columns = ['Value0', 'Value1', 'Rollout', 'Runtime', 'Method']
    columns = ['Value0', 'Value1', 'Rollout', 'Runtime', 'Method', 'Repetitions', 'Perturbation']
    df = pd.DataFrame(results, columns=columns)
    # df.to_csv(f'{path_data}results_{opt_str}_{method}_{file}_exp{args.exp_seed}_reps{args.reps}.csv', index=False)
    df.to_csv(f'{path_data}ND_results_all_{method}_{file}_exp{args.exp_seed}_{batch}_reps{args.reps}.csv', index=False)
