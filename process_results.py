import json
import argparse
import matplotlib
import pandas as pd
from utils import mkdir_p

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.despine()
sns.set_context("paper", rc={"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15,
                             "legend.fontsize": 16})
sns.set_style('white', {'axes.edgecolor': "0.5", "pdf.fonttype": 42})
plt.gcf().subplots_adjust(bottom=0.15, left=0.14)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-states', type=int, default=20, help="number of states")
    parser.add_argument('-obj', type=int, default=2, help="number of objectives")
    parser.add_argument('-act', type=int, default=3, help="number of actions")
    parser.add_argument('-suc', type=int, default=7, help="number of successors")
    parser.add_argument('-seed', type=int, default=42, help="seed")
    parser.add_argument('-exp_seed', type=int, default=2, help="experiment seed")
    parser.add_argument('-optimiser', type=str, default='nn', help="Optimiser")

    args = parser.parse_args()

    path_data = f'results/'
    path_plots = f'plots/'
    mkdir_p(path_plots)
    file = f'MPD_s{args.states}_a{args.act}_o{args.obj}_ss{args.suc}_seed{args.seed}_exp{args.exp_seed}'

    with open(f'{path_data}results_{args.optimiser}_{file}.json', "r") as read_file:
        info = json.load(read_file)
    print(info)

    results = pd.read_csv(f'{path_data}results_{args.optimiser}_{file}.csv')

    ax = sns.barplot(y='Runtime', data=results, ci='sd', label='NN')
    ax.set(ylabel='Average runtime (s)')
    plot_name = f"{path_plots}/{args.optimiser}_runtime"
    # plt.title(f"Action probabilities - Agent 2")
    plt.savefig(plot_name + ".pdf")

    plt.clf()


