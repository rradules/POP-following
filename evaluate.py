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


def select_action(state, pcs, objective_columns, value_vector):
    Q_next = pcs.loc[pcs['State'] == state]
    i_min = np.linalg.norm(Q_next[objective_columns] - value_vector, axis=1).argmin()
    action = Q_next['Action'].iloc[i_min]
    # print(str(value_vector)+","+str(action))
    return action


def get_value(state, action, pcs, objective_columns, value_vector):
    Q_next = pcs.loc[(pcs['State'] == state) & (pcs['Action'] == action)]
    i_min = np.linalg.norm(Q_next[objective_columns] - value_vector, axis=1).argmin()
    value = Q_next[objective_columns].iloc[i_min].values
    return value


def rollout(env, state0, action0, value_vector, pcs, gamma, max_time=200, optimiser=popf_local_search):
    # Assuming the state in the environment is indeed state0;
    # the reset needs to happen outside of this function
    time = 0
    stop = False
    action = action0
    state = state0
    returns = None
    cur_disc = 1
    while time < max_time and not stop:
        if value_vector is None:
            action = env.action_space.sample()
        # action picked, now let's execute it
        next_state, reward_vec, done, info = env.step(action)
        
        # keeping returns statistics:
        if returns is None:
            returns = cur_disc*reward_vec
        else: 
            returns += cur_disc*reward_vec
        # lowering the next timesteps forefactor:
        cur_disc *= gamma
        
        if value_vector is not None:
            n_vector = value_vector-reward_vec
            n_vector /= gamma
            next_probs = transition_function[state][action] # hacky, should be an argument
            problem = toStavs(next_probs, pcs)
            nm1, nm2, action, value_vector = optimiser(problem, n_vector, next_state)
        
        state = next_state
        stop = done
        time += 1
    return returns


def eval_POP_NN(env, s_prev, a_prev, v_prev):

    # Load the NN model

    d_in = num_objectives + 3
    d_out = num_objectives

    # TODO: layer size of the NN as argument?
    model = POP_NN([d_in, 16, 8, 4, d_out])
    model.load_state_dict(torch.load(f'{path_data}model_{file}.pth'))
    model.eval()
    ret_vector = np.zeros(num_objectives)
    cur_disc = 1

    done = False
    with torch.no_grad():
        while not done:
            s_next, r_next, done, _ = env.step(a_prev)
            ret_vector += cur_disc*r_next
            cur_disc *= gamma
            #print(s_prev, a_prev, s_next, r_next, done)
            N = (v_prev - r_next)/gamma
            inputNN = [s_prev / num_states, a_prev / num_actions, s_next / num_states]
            inputNN.extend(N)
            v_next = model.forward(torch.tensor(inputNN, dtype=torch.float32))[0].numpy()
            v_prev = v_next
                # v_next get_value(s_prev, a_prev, pcs, objective_columns, v_next)
            a_prev = select_action(s_next, pcs, objective_columns, v_next)
            s_prev = s_next


    #print(v0, ret_vector)
    return ret_vector


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-states', type=int, default=20, help="number of states")
    parser.add_argument('-obj', type=int, default=2, help="number of objectives")
    parser.add_argument('-act', type=int, default=3, help="number of actions")
    parser.add_argument('-suc', type=int, default=7, help="number of successors")
    parser.add_argument('-seed', type=int, default=42, help="seed")
    parser.add_argument('-exp_seed', type=int, default=2, help="experiment seed")
    parser.add_argument('-optimiser', type=str, default='ls', help="Optimiser")

    args = parser.parse_args()

    # reload environment

    register(
        id='RandomMOMDP-v0',
        entry_point='randommomdp:RandomMOMDP',
        reward_threshold=0.0,
        kwargs={'nstates': args.states, 'nobjectives': args.obj,
                'nactions': args.act, 'nsuccessor': args.suc, 'seed': args.seed}
    )

    np.random.seed(args.exp_seed)
    random.seed(args.exp_seed)
    torch.manual_seed(args.exp_seed)

    env = gym.make('RandomMOMDP-v0')
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    num_objectives = args.obj

    path_data = f'results/'
    file = f'MPD_s{args.states}_a{args.act}_o{args.obj}_ss{args.suc}_seed{args.seed}'

    num_states = args.states
    num_actions = args.act
    num_objectives = args.obj

    objective_columns = ['Objective 0', 'Objective 1']

    gamma = 0.8  # Discount factor

    with open(f'{path_data}{file}.json', "r") as read_file:
        env_info = json.load(read_file)

    transition_function = env_info['transition']
    reward_function = env_info['reward']

    pcs = pd.read_csv(f'{path_data}ND_PCS_{file}.csv')

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
    opt_str = 'nn'
    lsreps = 10

    if opt_str == 'nn':
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
            results.append(np.append(returns, [x, elapsed_seconds]))

        av = acc / times
        l = np.linalg.norm(v0 - av)
        print(f'NN: {l}, vec={av}')

    else:
        if opt_str == 'ls':
            optimiser = popf_local_search
        elif opt_str == 'mls':
            func = lambda a, b, c: popf_iter_local_search(a, b, c, reps=lsreps, pertrub_p=1)
            optimiser = func
        elif opt_str == 'ils':
            func = lambda a, b, c: popf_iter_local_search(a, b, c, reps=lsreps, pertrub_p=0.3)
            optimiser = func

        acc = np.array([0.0, 0.0])
        for x in range(times):
            start = time.time()
            env.reset()
            env._state = s0
            returns = rollout(env, s0, a0, v0, pcs, gamma, optimiser=optimiser)
            #print(f'{x+1}: {returns}', flush=True)
            end = time.time()
            elapsed_seconds = (end - start)
            acc = acc + returns
            results.append(np.append(returns, [x, elapsed_seconds]))

        av = acc/times
        l = np.linalg.norm(v0 - av)
        print(f'{opt_str}: {l}, vec={av}')

    final_result = {'method': opt_str, 'v0': v0.tolist()}
    json.dump(final_result, open(f'{path_data}results_{opt_str}_{file}_exp{args.exp_seed}.json', "w"))
    columns = ['Value0', 'Value1', 'Rollout', 'Runtime']
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(f'{path_data}results_{opt_str}_{file}_exp{args.exp_seed}.csv', index=False)




