import json
import matplotlib
import pandas as pd
from utils import mkdir_p, additive_epsilon_metric
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

    params = {'method': 'PQL', 'novec': 15, 'states': 110, 'obj': 2, 'act': 4, \
              'suc': 4, 'seed': 42, 'exp_seed': 1, 'opt': 'ils', 'reps': 10, 'batch': 128}

    path_data = f'results/'
    path_plots = f'plots/'
    mkdir_p(path_plots)
    file = f'{params["method"]}_s{params["states"]}_a{params["act"]}_o{params["obj"]}_' \
           f'ss{params["suc"]}_seed{params["seed"]}_novec{params["novec"]}_exp{params["exp_seed"]}'

    with open(f'{path_data}ND_results_all_{file}_{params["batch"]}_reps{params["reps"]}.json', "r") as read_file:
        info = json.load(read_file)
    v0 = info['v0']

    results = pd.read_csv(f'{path_data}ND_results_all_{file}_{params["batch"]}_reps{params["reps"]}.csv')
    results = results.loc[(results['Method'].isin(['nn','ls']) |
                           ((results['Method'] == 'mls') & (results['Repetitions'] == 10)) |
                           ((results['Method'] == 'ils') & (results['Repetitions'] == 10) &
                            (results['Perturbation'] == 0.3)))]

    for opt_str in ['nn', 'ls', 'mls', 'ils']: #['nn', 'ls', 'mls', 'ils']
        val_mean = results[['Value0', 'Value1']].loc[results['Method'] == opt_str].mean(axis=0).values
        val_diff = additive_epsilon_metric(val_mean, v0)
        print(opt_str, val_diff)
        #if params["states"] > 100:
        #    val_diff = multiplicative_epsilon_metric(val_mean, v0)
        #    print(opt_str, val_diff)
        #else:
        #    val_diff = additive_epsilon_metric(val_mean, v0)
        #    print(opt_str, val_diff)

    ###### PLOTS #########
    results.replace('nn', 'NN', inplace=True)
    results.replace('ls', 'LS', inplace=True)
    results.replace('ils', 'ILS', inplace=True)
    results.replace('mls', 'MLS', inplace=True)
    ax = sns.barplot(x='Method', y='Runtime', data=results, ci='sd')
    ax.set(ylabel='Average runtime (s)')
    ax.set_ylim(0, 15)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'),
                       (p.get_x() + p.get_width() / 4., p.get_height()+1.),
                       ha='center', va='center',
                       xytext=(0, 3),
                       textcoords='offset points')
    plot_name = f"{path_plots}/{file}_runtime2"
    plt.savefig(plot_name + ".pdf")
