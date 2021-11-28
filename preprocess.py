import argparse
import pandas as pd
from utils import get_non_dominated
import numpy as np
import gym
from gym.envs.registration import register


register(
        id='RandomMOMDP-v0',
        entry_point='randommomdp:RandomMOMDP',
)

register(
        id='DeepSeaTreasure-v0',
        entry_point='deep_sea_treasure:DeepSeaTreasureEnv',
)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-states', type=int, default=110, help="number of states")
    parser.add_argument('-obj', type=int, default=2, help="number of objectives")
    parser.add_argument('-act', type=int, default=4, help="number of actions")
    parser.add_argument('-suc', type=int, default=4, help="number of successors")
    parser.add_argument('-seed', type=int, default=42, help="seed")
    parser.add_argument('-env', type=str, default='RandomMOMDP-v0', help="The environment to run PVI on.")
    parser.add_argument('-noise', type=float, default=0.1, help="The stochasticity in state transitions.")
    parser.add_argument('-method', type=str, default='PQL', help="method")
    parser.add_argument('-novec', type=int, default=10, help="number of vectors")

    args = parser.parse_args()
    method = args.method

    path_data = f'results/'
    file = f's{args.states}_a{args.act}_o{args.obj}_ss{args.suc}_seed{args.seed}_novec{args.novec}'

    pcs = pd.read_csv(f'{path_data}PCS_{args.method}_{file}.csv')

    objective_columns = ['Objective 0', 'Objective 1']
    pcs[objective_columns] = pcs[objective_columns].apply(pd.to_numeric)
    non_dom_data = []
    env_name = args.env

    if args.states < 100:
        env = gym.make('RandomMOMDP-v0', nstates=args.states, nobjectives=args.obj, nactions=args.act, nsuccessor=args.suc, seed=args.seed)
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        num_objectives = env._nobjectives
        num_successors = args.suc
        transition_function = env._transition_function
        reward_function = env._old_reward_function
    else:
        env = gym.make('DeepSeaTreasure-v0', seed=args.seed, noise=args.noise)
        num_states = env.nS
        num_actions = env.nA
        num_objectives = 2
        num_successors = env.nS
        transition_function = env._transition_function
        reward_function = env._reward_function

    # for s in range(args.states):
    #     for a in range(args.act):
    #         subset = pcs.loc[(pcs['State'] == s) & (pcs['Action'] == a)]
    #         cand = subset[objective_columns].to_numpy()
    #         non_dom = get_non_dominated(cand)
    #
    #         for el in non_dom:
    #             non_dom_entry = pcs.loc[(pcs['Objective 0'] == el[0]) & (pcs['Objective 1'] == el[1])].iloc[0]
    #             non_dom_data.append(non_dom_entry)
    #
    # df = pd.concat(non_dom_data, axis=1).T
    # df.to_csv(f'{path_data}ND_PCS_{method}_{file}.csv', index=False)

    nn = pd.read_csv(f'{path_data}NN_{method}_{file}.csv')
    val_columns = ['vs0', 'vs1']
    nn[val_columns] = nn[val_columns].apply(pd.to_numeric)
    non_dom_data = []

    for s in range(args.states):
        for a in range(args.act):
            #next_states = np.where(transition_function[s, a, :] > 0)[0]
            for ns in range(args.states):
                subset = nn.loc[(nn['s'] == s) & (nn['a'] == a) & (nn['ns'] == ns)]
                cand = subset[val_columns].to_numpy()
                if len(cand) > 0:
                    non_dom = get_non_dominated(cand)

                    for el in non_dom:
                        non_dom_entry = nn.loc[(nn['vs0'] == el[0]) & (nn['vs1'] == el[1])].iloc[0]
                    non_dom_data.append(non_dom_entry)

    df = pd.concat(non_dom_data, axis=1).T
    df.to_csv(f'{path_data}ND_NN_{method}_{file}.csv', index=False)