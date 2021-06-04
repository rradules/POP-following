import argparse
import pandas as pd
from utils import get_non_dominated

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-states', type=int, default=10, help="number of states")
    parser.add_argument('-obj', type=int, default=2, help="number of objectives")
    parser.add_argument('-act', type=int, default=2, help="number of actions")
    parser.add_argument('-suc', type=int, default=4, help="number of successors")
    parser.add_argument('-seed', type=int, default=777, help="seed")

    args = parser.parse_args()

    path_data = f'results/'
    file = f'MPD_s{args.states}_a{args.act}_o{args.obj}_ss{args.suc}_seed{args.seed}'

    pcs = pd.read_csv(f'{path_data}PCS_{file}.csv')

    objective_columns = ['Objective 0', 'Objective 1']
    pcs[objective_columns] = pcs[objective_columns].apply(pd.to_numeric)
    non_dom_data = []

    for s in range(args.states):
        subset = pcs.loc[pcs['State'] == s]
        cand = subset[objective_columns].to_numpy()
        non_dom = get_non_dominated(cand)

        for el in non_dom:
            non_dom_entry = pcs.loc[(pcs['Objective 0'] == el[0]) & (pcs['Objective 1'] == el[1])].iloc[0]
            non_dom_data.append(non_dom_entry)

    df = pd.concat(non_dom_data, axis=1).T
    df.to_csv(f'{path_data}ND_PCS_{file}.csv', index=False)

    nn = pd.read_csv(f'{path_data}NN_{file}.csv')
    val_columns = ['vs0', 'vs1']
    nn[val_columns] = nn[val_columns].apply(pd.to_numeric)
    non_dom_data = []

    for s in range(args.states):
        subset = nn.loc[nn['s'] == s]
        cand = subset[val_columns].to_numpy()
        non_dom = get_non_dominated(cand)

        for el in non_dom:
            non_dom_entry = nn.loc[(nn['vs0'] == el[0]) & (nn['vs1'] == el[1])].iloc[0]
            non_dom_data.append(non_dom_entry)

    df = pd.concat(non_dom_data, axis=1).T
    df.to_csv(f'{path_data}ND_NN_{file}.csv', index=False)