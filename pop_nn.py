import time
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from networks import Mlp, MlpLarge, MlpSmall, MlpWithBatchNorm

from sklearn.model_selection import train_test_split


def one_hot_encode(data):
    one_hot_encoded = []
    state_idx = 0
    action_idx = 1
    next_state_idx = 2

    for row in data:
        row = list(row)
        state = int(row[state_idx])
        action = int(row[action_idx])
        next_state = int(row[next_state_idx])
        N = row[-num_objectives:]
        one_hot_state = list(np.zeros(num_states))
        one_hot_action = list(np.zeros(num_actions))
        one_hot_next_state = list(np.zeros(num_states))
        one_hot_state[state] = 1
        one_hot_action[action] = 1
        one_hot_next_state[next_state] = 1

        new_row = one_hot_state + one_hot_action + one_hot_next_state + N
        one_hot_encoded.append(new_row)

    return one_hot_encoded


def preprocess_data(data_file, num_objectives, batch, normalise=False, num_states=110, num_actions=4):
    """
    This function preprocesses the data and loads it into a train and validation loader.
    :param data_file: The raw csv file path.
    :param num_objectives: The number of objectives. Determines the targets for the neural network.
    :param batch: The batch size to use in the data loaders.
    :param normalise: Whether to normalise the data or not.
    :param num_states: The number of states. Used for normalisation.
    :param num_actions: The number of actions. Used for normalisation.
    :return:
    """
    data = pd.read_csv(data_file)  # Load training data.

    if normalise:  # Normalise states and actions.
        data['s'] = data['s'].values / num_states
        data['ns'] = data['ns'].values / num_states
        data['a'] = data['a'].values / num_actions

    # Separate train from target (changes here is action should be a target)
    target = data[data.columns[-num_objectives:]].values
    train = data[data.columns[:-num_objectives]].values
    train_processed = []
    action_idx = 1

    # train = one_hot_encode(train)

    # 80-20 train - validation
    X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2)

    # Create the data loaders
    train = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.float32))
    val = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                        torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train, shuffle=True, batch_size=batch)
    val_loader = DataLoader(val, shuffle=True, batch_size=batch)

    return train_loader, val_loader


def load_network(model_str, num_objectives, dropout=0.5, checkpoint=None):
    """
    This function loads a neural network with the required parameters.
    :param model_str: The name of the neural network model to use.
    :param num_objectives: The number of objectives. This determines the number of inputs and outputs in the network.
    :param dropout: The dropout rate in the network.
    :param checkpoint: The file containing the last training checkpoint.
    :return:
    """
    d_in = num_objectives + 3
    d_out = num_objectives

    if model_str == 'Mlp':
        model = Mlp(d_in, d_out, dropout)
    elif model_str == 'MlpLarge':
        model = MlpLarge(d_in, d_out, dropout)
    elif model_str == 'MlpSmall':
        model = MlpSmall(d_in, d_out, dropout)
    elif model_str == 'MlpBatchnorm':
        model = MlpWithBatchNorm(d_in, d_out, dropout)
    else:
        raise Exception

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
    return model


def train_pop_network(model, train_loader, val_loader, output_file, epochs=1000, predict_every=20):
    """
    This function trains a neural network.
    :param model: The neural network.
    :param train_loader: The loader for the training data.
    :param val_loader: The loader for the validation data.
    :param output_file: The filepath for saving the network parameters.
    :param epochs: The total number of epochs to train for.
    :param predict_every: Test on the validation set when this number of epochs has passed.
    :return: The loss of the best model, the loss over time and the final model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print(f'Model running on {device}')

    best_loss = float('inf')
    loss_over_time = []  # We can save this for plotting purposes.
    start = time.time()

    for epoch in range(epochs):
        print(f'Running epoch {epoch}')
        epoch_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += output.shape[0] * loss.item()

        total_loss = epoch_loss / len(train_loader.dataset)
        loss_over_time.append(total_loss)
        print(f'Training loss is: {total_loss}')

        if epoch % predict_every == 0:
            model.eval()
            validation_loss = 0

            for batch_idx, (data, target) in enumerate(val_loader):
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                loss = loss_function(output, target)
                validation_loss += output.shape[0] * loss.item()

            total_validation_loss = validation_loss / len(val_loader.dataset)
            if total_validation_loss < best_loss:
                best_loss = total_validation_loss
                torch.save(model.state_dict(), output_file)
                print(f'Saving new model with validation loss: {best_loss}')
            else:
                print(f'Validation loss did not improve. Best model is still at: {best_loss}')
            model.train()

        print('-------------------------------------------')

    end = time.time()
    elapsed_mins = (end - start) / 60.0
    print(f'Finished training after {elapsed_mins} minutes')
    print(f'Best model performance reached loss: {best_loss}')

    return best_loss, loss_over_time, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', type=str, default='results/NN_PVI_s110_a4_o2_ss4_seed42_novec10.csv', help='The path to the file containing the training data')
    parser.add_argument('-output', type=str, default='results/model.pth', help='The file for saving the network.')
    parser.add_argument('-checkpoint', type=str, default=None, help='A pretrained network to finetune.')
    parser.add_argument('-states', type=int, default=110, help="number of states")
    parser.add_argument('-act', type=int, default=4, help="number of actions")
    parser.add_argument('-obj', type=int, default=2, help="number of objectives")
    parser.add_argument('-model', type=str, default='MlpSmall', help="The network architecture to use.")
    parser.add_argument('-normalise', type=bool, default=False, help='Normalise input data')
    parser.add_argument('-epochs', type=int, default=3000, help="epochs")
    parser.add_argument('-batch', type=int, default=1024, help="batch size")
    parser.add_argument('-dropout', type=float, default=0., help='Dropout rate for the neural network')

    args = parser.parse_args()

    # Extract arguments.
    data_file = args.data
    output_file = args.output
    checkpoint_file = args.checkpoint
    num_states = args.states
    num_actions = args.act
    num_objectives = args.obj
    model_str = args.model
    normalise = args.normalise
    epochs = args.epochs
    batch = args.batch
    dropout = args.dropout

    # Load the data and network and train it.
    train_loader, val_loader = preprocess_data(data_file, num_objectives, batch, normalise=normalise, num_states=num_states, num_actions=num_actions)
    model = load_network(model_str, num_objectives, dropout, checkpoint=checkpoint_file)
    best_loss, loss_over_time, model = train_pop_network(model, train_loader, val_loader, output_file, epochs=epochs)

