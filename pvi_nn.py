import gym
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

from gym.envs.registration import register


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
        super(MLP, self).__init__()
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
    register(
        id='RandomMOMDP-v0',
        entry_point='randommomdp:RandomMOMDP',
        reward_threshold=0.0,
        kwargs={'nstates': 5, 'nobjectives': 2, 'nactions': 2, 'nsuccessor': 3, 'seed': 1}
    )

    env = gym.make('RandomMOMDP-v0')
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    num_objectives = env._nobjectives


    #TODO: load environemnt params and dataset, train and evaluate network
    d_in = 36
    d_out = 48
    model = POP_NN([d_in, 128, 64, d_out]).to(device)

    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    #criterion = nn.NLLLoss()

    #model, losses, accuracies = train_val_model(model, criterion, optimizer, dataloaders,
    # num_epochs=10, log_interval=2)
