import json
import matplotlib
import pandas as pd
from utils import mkdir_p
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set()
sns.despine()
sns.set_context("paper", rc={"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15,
                             "legend.fontsize": 16})
sns.set_style('white', {'axes.edgecolor': "0.5", "pdf.fonttype": 42})
plt.gcf().subplots_adjust(bottom=0.15, left=0.14)


if __name__ == '__main__':

    params = {'states': 20, 'obj': 2, 'act': 3, 'suc': 7, 'seed': 42, 'exp_seed': 2, 'opt': 'ils', 'reps': 10}

    path_data = f'results/'
    path_plots = f'plots/'
    mkdir_p(path_plots)
    file = f'MPD_s{params["states"]}_a{params["act"]}_o{params["obj"]}_' \
           f'ss{params["suc"]}_seed{params["seed"]}_exp{params["exp_seed"]}'

    with open(f'{path_data}results_{params["opt"]}_{file}.json', "r") as read_file:
        info = json.load(read_file)
    v0 = info['v0']
    results = pd.read_csv(f'{path_data}results_{params["opt"]}_{file}_reps{params["reps"]}.csv')

    for opt_str in ['nn', 'ls', 'mls', 'ils']:
        val_mean = results[['Value0', 'Value1']].loc[results['Method'] == opt_str].mean(axis=0).values
        val_diff = v0 - val_mean
        print(opt_str, max(val_diff))

    '''
    ax = sns.barplot(x='Method', y='Runtime', data=results, ci='sd')
    ax.set(ylabel='Average runtime (s)')
    plot_name = f"{path_plots}/{file}_runtime"
    # plt.title(f"Action probabilities - Agent 2")
    plt.savefig(plot_name + ".pdf")

    plt.clf()
    '''



