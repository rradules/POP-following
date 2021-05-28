import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from gym.envs.registration import register
import json
import argparse
import pandas as pd
from pvi import get_non_dominated
from pop_nn import POP_NN
from utils import is_dominated
import numpy as np
import random

def eval_POP_NN(env, s_prev, a_prev, v_prev):

    # Load the NN model

    d_in = num_objectives + 3
    d_out = num_objectives

    # TODO: layer size of the NN as argument?
    model = POP_NN([d_in, 8, 4, d_out])
    model.load_state_dict(torch.load(f'{path_data}model_{file}.pth'))
    model.eval()

    done = False
    with torch.no_grad():
        while not(done):
            s_next, r_next, done, _ = env.step(a_prev)
            print(s_prev, a_prev, s_next, r_next, done)
            N = (v_prev - r_next)/gamma
            input = [s_prev / num_states, a_prev / num_actions, s_next / num_states]
            input.extend(N)
            v_next = model.forward(torch.tensor(input, dtype=torch.float32))[0].numpy()
            Q_next = pcs.loc[pcs['State'] == s_next]
            objective_columns = ['Objective 0', 'Objective 1']
            i_min = np.linalg.norm(Q_next[objective_columns] - v_next, axis=1).argmin()
            a_prev = Q_next['Action'].iloc[i_min]
            s_prev = s_next
            v_prev = v_next

    print(v0, v_next)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-states', type=int, default=10, help="number of states")
    parser.add_argument('-obj', type=int, default=2, help="number of objectives")
    parser.add_argument('-act', type=int, default=2, help="number of actions")
    parser.add_argument('-suc', type=int, default=4, help="number of successors")
    parser.add_argument('-seed', type=int, default=1, help="seed")

    parser.add_argument('-exp_seed', type=int, default=42, help="experiment seed")

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
    num_objectives = env._nobjectives

    transition_function = env._transition_function
    reward_function = env._reward_function

    path_data = f'results/'
    file = f'MPD_s{args.states}_a{args.act}_o{args.obj}_ss{args.suc}_seed{args.seed}'


    num_states = args.states
    num_actions = args.act
    num_objectives = args.obj

    gamma = 0.8  # Discount factor

    with open(f'{path_data}{file}.json', "r") as read_file:
        env_info = json.load(read_file)

    pcs = pd.read_csv(f'{path_data}PCS_{file}.csv')

    pcs[['Objective 0', 'Objective 1']] = pcs[['Objective 0', 'Objective 1']].apply(pd.to_numeric)

    s0 = env.reset()
    dom = True
    subset = pcs[['Action', 'Objective 0', 'Objective 1']].loc[pcs['State'] == s0]
    cand = [subset[['Objective 0', 'Objective 1']].to_numpy()]


    # Select initial non-dominated value
    while dom:
        select = subset.sample()
        a0 = select['Action'].iloc[0]
        v0 = select[['Objective 0', 'Objective 1']].iloc[0].values
        dom = is_dominated(v0, cand)

    print(s0, a0, v0)
    eval_POP_NN(env, s0, a0, v0)



