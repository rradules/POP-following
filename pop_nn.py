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

# Device configuration
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu' # Force CPU
print(device)


class POP_NN(nn.Module):
    """
    Simple MLP model
    :param d_layer: array with the layer size configuration
    """

    def __init__(self, d_layer):
        super(POP_NN, self).__init__()
        self.d_layer = d_layer
        layer_list = [nn.Linear(d_layer[l], d_layer[l + 1]) for l in range(len(d_layer) - 1)]
        self.lins = nn.ModuleList(layer_list)

    def forward(self, x):
        x = x.view(-1, self.d_layer[0])
        for layer in self.lins[:-1]:
            x = F.relu(layer(x))
        return self.lins[-1](x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-states', type=int, default=110, help="number of states")
    parser.add_argument('-obj', type=int, default=2, help="number of objectives")
    parser.add_argument('-act', type=int, default=4, help="number of actions")
    parser.add_argument('-suc', type=int, default=4, help="number of successors")
    parser.add_argument('-seed', type=int, default=42, help="seed")
    parser.add_argument('-method', type=str, default='PQL', help="method")
    parser.add_argument('-novec', type=int, default=10, help="number of vectors")
    parser.add_argument('-batch', type=int, default=8, help="batch size")
    parser.add_argument('-nnl', help='NN layer structure', type=lambda s: [int(item) for item in s.split(',')])

    args = parser.parse_args()

    path_data = f'results/'
    file = f's{args.states}_a{args.act}_o{args.obj}_ss{args.suc}_seed{args.seed}_novec{args.novec}'

    num_states = args.states
    num_actions = args.act
    num_objectives = args.obj
    method = args.method
    batch = args.batch
    nnl = args.nnl

    # Load training data
    data = pd.read_csv(f'{path_data}ND_NN_{method}_{file}.csv')

    # normalisation for states and actions
    data['s'] = data['s'].values / num_states
    data['ns'] = data['ns'].values / num_states
    data['a'] = data['a'].values / num_actions

    # Separate train from target (changes here is action should be a target)
    target = data[data.columns[-num_objectives:]].values
    train = data[data.columns[:-num_objectives]].values

    # 80-20 train - validation
    X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2)

    # Create the data loaders
    train = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.float32))  # create your datset
    val = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                        torch.tensor(y_val, dtype=torch.float32))  # create your datset

    train_loader = DataLoader(train, shuffle=True, batch_size=batch)  # create your dataloader
    val_loader = DataLoader(val, shuffle=True, batch_size=batch)  # create your dataloader

    # input output size
    d_in = num_objectives + 3
    d_out = num_objectives
    # init NN
    nnl.insert(0, d_in)
    nnl.append(d_out)
    model = POP_NN(nnl).to(device)
    loss_function = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    n_epochs = 3000
    predict_every = 20
    min_valid_loss = 1.0
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            #if torch.cuda.is_available():
             #   data, labels = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            train_loss = loss.data
            # if batch_idx % predict_every == 0:
            #    print(f'Train Epoch: {epoch}, Loss: {loss.data}')

        model.eval()  # Optional when not using Model Specific layer
        for batch_idx, (data, target) in enumerate(val_loader):
            #if torch.cuda.is_available():
            #    data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = loss_function(output, target)
            valid_loss = loss.data
            # if batch_idx % predict_every == 0:
            #    print(f'Epoch {epoch} \t\t Training Loss: {train_loss } '
            #      f'\t\t Validation Loss: {valid_loss }')

            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss
                # Saving State Dict
                torch.save(model.state_dict(), f'{path_data}ND_model_{batch}_{method}_{file}.pth')
