import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from gym.envs.registration import register
import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

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
        self.lins = nn.ModuleList(layer_list)

    def forward(self, x):
        x = x.view(-1, self.d_layer[0])
        for layer in self.lins[:-1]:
            x = F.relu(layer(x))
        return self.lins[-1](x)
        #F.softmax(self.lins[-1](x), dim=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-states', type=int, default=10, help="number of states")
    parser.add_argument('-obj', type=int, default=2, help="number of objectives")
    parser.add_argument('-act', type=int, default=2, help="number of actions")
    parser.add_argument('-suc', type=int, default=4, help="number of successors")
    parser.add_argument('-seed', type=int, default=2, help="seed")

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

    data = pd.read_csv(f'{path_data}NN_{file}.csv')

    data['s'] = data['s'].values/num_states
    data['ns'] = data['ns'].values / num_states
    data['a'] = data['a'].values / num_actions

    target = data[data.columns[-num_objectives:]].values
    train = data[data.columns[:-num_objectives]].values

    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    train = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.float32))  # create your datset
    val = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                        torch.tensor(y_val, dtype=torch.float32))  # create your datset
    test = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                         torch.tensor(y_test, dtype=torch.float32))  # create your datset

    train_loader = DataLoader(train, shuffle=True, batch_size=16)  # create your dataloader
    val_loader = DataLoader(val, shuffle=True, batch_size=16)  # create your dataloader
    test_loader = DataLoader(test, shuffle=True, batch_size=16)  # create your dataloader

    #TODO: train and evaluate network
    d_in = num_objectives + 3
    d_out = num_objectives
    model = POP_NN([d_in, 8, 4, d_out]).to(device)
    model = model.float()

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    #criterion = nn.NLLLoss()

    n_epochs = 1000
    predict_every = 20
    min_valid_loss = 0.05
    for epoch in range(n_epochs):

        #for local_batch, local_labels in train_loader:
        for batch_idx, (data, target) in enumerate(train_loader):
            if torch.cuda.is_available():
                data, labels = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            train_loss = loss.data
            #if batch_idx % predict_every == 0:
            #    print(f'Train Epoch: {epoch}, Loss: {loss.data}')

        model.eval()  # Optional when not using Model Specific layer
        for batch_idx, (data, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = loss_function(output, target)
            valid_loss = loss.data
            #if batch_idx % predict_every == 0:
            #    print(f'Epoch {epoch} \t\t Training Loss: {train_loss } '
            #      f'\t\t Validation Loss: {valid_loss }')

            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss
                # Saving State Dict
                torch.save(model.state_dict(), f'{path_data}model_{file}.pth')
