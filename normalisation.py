import argparse
import pandas as pd
import json
from utils import get_non_dominated
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-states', type=int, default=110, help="number of states")
    parser.add_argument('-obj', type=int, default=2, help="number of objectives")
    parser.add_argument('-act', type=int, default=4, help="number of actions")
    parser.add_argument('-suc', type=int, default=4, help="number of successors")
    parser.add_argument('-seed', type=int, default=42, help="seed")
    parser.add_argument('-method', type=str, default='PVI', help="method")
    parser.add_argument('-novec', type=int, default=10, help="number of vectors")

    args = parser.parse_args()
    method = args.method

    path_data = f'results/'
    file = f's{args.states}_a{args.act}_o{args.obj}_ss{args.suc}_seed{args.seed}_novec{args.novec}'

    data = pd.read_csv(f'{path_data}ND_NN_{method}_{file}.csv')
    d_min = data[['N0', 'N1', 'vs0', 'vs1']].min().min()
    d_max = data[['N0', 'N1', 'vs0', 'vs1']].max().max()

    data.loc[:, ['N0', 'N1', 'vs0', 'vs1']] = (data.loc[:, ['N0', 'N1', 'vs0', 'vs1']] - d_min) / (d_max - d_min)

    data.to_csv(f'{path_data}ND_normNN_{method}_{file}.csv', index=False)

    dict_norm = {'min':d_min, 'max':d_max}

    with open(f'{path_data}ND_normNN_{method}_{file}.json', 'w') as f:
        json.dump(dict_norm, f)
