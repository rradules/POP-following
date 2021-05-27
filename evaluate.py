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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-states', type=int, default=10, help="number of states")
    parser.add_argument('-obj', type=int, default=2, help="number of objectives")
    parser.add_argument('-act', type=int, default=2, help="number of actions")
    parser.add_argument('-suc', type=int, default=4, help="number of successors")
    parser.add_argument('-seed', type=int, default=1, help="seed")

    args = parser.parse_args()

    # reload environment

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


    path_data = f'results/'
    file = f'MPD_s{args.states}_a{args.act}_o{args.obj}_ss{args.suc}_seed{args.seed}'

    num_states = args.states
    num_actions = args.act
    num_objectives = args.obj

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
        select = subset.sample().values[0]
        a0 = select[0]
        v0 = select[1:]
        dom = is_dominated(v0, cand)

    print(s0, a0, v0)


    '''
    # Load the NN model

    d_in = num_objectives + 3
    d_out = num_objectives

    # TODO: layer size of the NN as argument?
    model = POP_NN([d_in, 8, 4, d_out])
    model.load_state_dict(torch.load(f'{path_data}model_{file}.pth'))
    model.eval()

    '''


