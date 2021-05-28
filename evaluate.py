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
import copy

def select_action(state, pcs, value_vector):
    return 0 #TODO: stub

def rollout(env, state0, action0, value_vector, pcs, gamma, max_time=1000, value_selector=None):
    #Assuming the state in the environment is indeed state0;
    #the reset needs to happen outside of this function
    time = 0
    stop = False
    action = action0
    state  = state0
    returns = None
    cur_disc = 1
    while(time<max_time and not stop):
        if (value_vector is not None) and (action is None) :
            action = select_action(state, pcs, value_vector)
        else:
            action = env.action_space.sample()
        #action picked, now let's execute it
        observation, reward_vec, done, info = env.step(action)
        
        #keeping returns statistics:
        if(returns is None):
            returns=cur_disc*reward_vec
        else: 
            returns += cur_disc*reward_vec
        #lowering the next timesteps forefactor:
        cur_disc*=gamma
        
        if value_vector is not None:
            n_vector = value_vector-reward_vec
            n_vector /= gamma
            print(n_vector) 
            value_vector=None #TODO:replace
            
        action=None
        stop=done
        time+=1
        
    return returns

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
        while not done:
            s_next, r_next, done, _ = env.step(a_prev)
            print(s_prev, a_prev, s_next, r_next, done)
            N = (v_prev - r_next)/gamma
            inputNN = [s_prev / num_states, a_prev / num_actions, s_next / num_states]
            inputNN.extend(N)
            v_next = model.forward(torch.tensor(inputNN, dtype=torch.float32))[0].numpy()
            Q_next = pcs.loc[pcs['State'] == s_next]
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

    objective_columns = ['Objective 0', 'Objective 1']

    gamma = 0.8  # Discount factor

    with open(f'{path_data}{file}.json', "r") as read_file:
        env_info = json.load(read_file)

    pcs = pd.read_csv(f'{path_data}PCS_{file}.csv')

    pcs[objective_columns] = pcs[objective_columns].apply(pd.to_numeric)

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
    returns = rollout(env, s0, a0, v0, pcs, 0.8)
    print(returns)
    eval_POP_NN(env, s0, a0, v0)



