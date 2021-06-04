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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-states', type=int, default=10, help="number of states")
    parser.add_argument('-obj', type=int, default=2, help="number of objectives")
    parser.add_argument('-act', type=int, default=2, help="number of actions")
    parser.add_argument('-suc', type=int, default=4, help="number of successors")
    parser.add_argument('-seed', type=int, default=42, help="seed")

    args = parser.parse_args()

    path_data = f'results/'
    file = f'MPD_s{args.states}_a{args.act}_o{args.obj}_ss{args.suc}_seed{args.seed}'

    pcs = pd.read_csv(f'{path_data}PCS_{file}.csv')

    objective_columns = ['Objective 0', 'Objective 1']
    pcs[objective_columns] = pcs[objective_columns].apply(pd.to_numeric)