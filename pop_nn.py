import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.envs.registration import register
import json
import argparse
import pandas as pd
import ast
import numpy as np

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu' # Force CPU
print(device)


class POP_NN(nn.Module):
    """
    Simple MLP model
    :param d_layer: array with the layer size configuration
    """
    def __init__(self, d_layer):
        super(POP_NN, self).__init__()
        self.d_layer = d_layer
        layer_list = [nn.Linear(d_layer[l], d_layer[l+1]) for l in range(len(d_layer) - 1)]
        self.linears = nn.ModuleList(layer_list)

    def forward(self, x):
        x = x.view(-1, self.d_layer[0])
        # relu(Wl x) for all hidden layer
        for layer in self.linears[:-1]:
            x = F.relu(layer(x))
        # softmax(Wl x) for output layer
        return F.log_softmax(self.linears[-1](x), dim=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-states', type=int, default=10, help="number of states")
    parser.add_argument('-obj', type=int, default=2, help="number of objectives")
    parser.add_argument('-act', type=int, default=2, help="number of actions")
    parser.add_argument('-suc', type=int, default=4, help="number of successors")
    parser.add_argument('-seed', type=int, default=1, help="seed")

    args = parser.parse_args()

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
    file = f'MPD_s{num_states}_a{num_actions}_o{num_objectives}_ss{args.suc}_seed{args.seed}'

    with open(f'{path_data}{file}.json', "r") as read_file:
        env_info = json.load(read_file)

    #TODO: load environemnt params and dataset, train and evaluate network
    d_in = num_objectives + 3
    d_out = num_objectives + 1
    model = POP_NN([d_in, 128, 64, d_out]).to(device)

    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    #criterion = nn.NLLLoss()

    #model, losses, accuracies = train_val_model(model, criterion, optimizer, dataloaders,
    # num_epochs=10, log_interval=2)

    #TODO: Did not do % slit in  test train

    data = pd.read_csv(f'{path_data}NN_{file}.csv')

    target = torch.tensor(data[data.columns[-num_objectives:]].values)
    train = torch.tensor(data[data.columns[:-num_objectives]].values)