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

class Stav:
    def __init__(self, state, probab, action=None, vector=None):
        self.state = state
        self.prob = probab
        self.action = action
        self.vector = vector
        self.tvector = self.prob*vector if vector is not None else None
        self.pcs_table = (pcs.loc[pcs['State'] == state]) #hacky, should be an argument
        self.pcs_list = self.precompute_tuples()
        self.pprune_tuples()
        self.pick_random()
        
    def __str__(self):
        return "("+str(self.state)+","+str(self.prob)+","+str(self.action)+","+str(self.vector)+","+str(self.tvector)+")"
        
    def pick_random(self):
        row = self.pcs_table.sample().to_numpy()[0]
        self.action = int(row[1])
        self.vector = row[2:]
        self.tvector = self.prob*self.vector
        
    def pprune_tuples(self):
        new_tuples = []
        while(len(self.pcs_list)>0):
            nn_tups = []
            ctup = self.pcs_list[0]
            for tup in self.pcs_list:
                if(np.greater_equal(tup[1], ctup[1]).all()):
                    ctup = tup
            for tup in self.pcs_list:
                if(not np.greater_equal(ctup[1], tup[1]).all()):
                    nn_tups.append(tup)
            new_tuples.append(ctup)
            self.pcs_list = nn_tups
        self.pcs_list = new_tuples
        
    def precompute_tuples(self):
        result = []
        for index, row in self.pcs_table.iterrows():
            line = row.to_numpy()
            act  = int(line[1])
            vec  = line[2:]
            tvec = self.prob * vec
            result.append( (act,vec,tvec) )
        return result
        
    def improve_step(self, n_vector, cur_vector, cur_score = None):
        cur_score = np.linalg.norm(n_vector - cur_vector) if cur_score is None else cur_score
        min_vec = n_vector - cur_vector + self.tvector
        nw_cur_vector = cur_vector - self.tvector
        for tup in self.pcs_list:
            n_score = np.linalg.norm(min_vec + tup[2])
            if(n_score < cur_score):
                self.action = tup[0]
                self.vector = tup[1]
                self.tvector= tup[2]
                cur_score = n_score
                return True, (nw_cur_vector+self.tvector), cur_score
        return False, cur_vector, cur_score
        
        
    
def toStavs(trans):
    result = []
    for i in range(len(trans)):
        if(trans[i]>0):
            tup = Stav(i,trans[i])
            result.append(tup)
    return result

def select_action(state, pcs, value_vector):
    Q_next = pcs.loc[pcs['State'] == state]
    i_min = np.linalg.norm(Q_next[objective_columns] - value_vector, axis=1).argmin()
    action = Q_next['Action'].iloc[i_min]
    #print(str(value_vector)+","+str(action)) 
    return action
    
def popf_local_search(problem, n_vector, state, cur_vector = None, score = None):
    if(cur_vector is None or score is None):
        for stav in problem:
            cur_vector = stav.tvector if cur_vector is None else cur_vector+stav.tvector
        score = np.linalg.norm(n_vector - cur_vector)
    improved = True
    while(improved): 
        improved = False
        random.shuffle(problem)
        for stav in problem:
            improved, cur_vector, score = stav.improve_step(n_vector, cur_vector, score)
            if improved:
                break
    for stav in problem:
        if stav.state == state:
            action = stav.action
            value_vector = stav.vector            
    return cur_vector, score, action, value_vector
    
def popf_iter_local_search(problem, n_vector, state, reps=10, pertrub_p=1.0):
    cur_vector=None
    for stav in problem:
        stav.pick_random()
        cur_vector = stav.tvector if cur_vector is None else cur_vector+stav.tvector
        if stav.state == state:
            action = stav.action
            value_vector = stav.vector
    score = np.linalg.norm(n_vector - cur_vector)
    for i in range(reps):
        for stav in problem:
            rnumber = random.random()
            if(rnumber<pertrub_p):
                stav.pick_random()
        cur_vector_rep, score_rep, action_rep, value_vector_rep = popf_local_search(problem, n_vector, state)
        if(score_rep < score):
            cur_vector = cur_vector_rep
            score = score_rep
            action = action_rep
            value_vector = value_vector_rep
    return cur_vector, score, action, value_vector        
        
    

def rollout(env, state0, action0, value_vector, pcs, gamma, max_time=200, optimiser=popf_local_search):
    #Assuming the state in the environment is indeed state0;
    #the reset needs to happen outside of this function
    time = 0
    stop = False
    action = action0
    state  = state0
    returns = None
    cur_disc = 1
    while(time<max_time and not stop):
        if (value_vector is None):
            action = env.action_space.sample()
        #action picked, now let's execute it
        next_state, reward_vec, done, info = env.step(action)
        
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
            next_probs = transition_function[state][action] #hacky, should be an argument
            problem = toStavs(next_probs)
            nm1, nm2, action, value_vector = optimiser(problem, n_vector, next_state)
        
        state=next_state    
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
    acc = np.array([0.0,0.0])
    times=25
    for x in range(times):
        env.reset()
        env._state = s0
        returns = rollout(env, s0, a0, v0, pcs, 0.8)
        print(str(x+1)+":"+str(returns), end=" ", flush=True)
        acc = acc+returns
    av = acc/times
    l = np.linalg.norm(v0 - av)
    print("\nls: "+str(l)+" vec="+str(av))
    acc = np.array([0.0,0.0])
    for x in range(times):
        env.reset()
        env._state = s0
        returns = rollout(env, s0, a0, v0, pcs, 0.8, optimiser=popf_iter_local_search)
        print(str(x+1)+": "+str(returns), end=" ", flush=True)
        acc = acc+returns
    av = acc/times
    l = np.linalg.norm(v0 - av)
    print("\nmls: "+str(l)+" vec="+str(av))
    acc = np.array([0.0,0.0])
    for x in range(times):
        env.reset()
        env._state = s0
        func = lambda a, b, c : popf_iter_local_search(a,b,c,reps=10, pertrub_p=0.3)
        returns = rollout(env, s0, a0, v0, pcs, 0.8, optimiser=popf_iter_local_search)
        print(str(x+1)+": "+str(returns), end=" ", flush=True)
        acc = acc+returns
    av = acc/times
    l = np.linalg.norm(v0 - av)
    print("\nils: "+str(l)+" vec="+str(av))

    #eval_POP_NN(env, s0, a0, v0)



