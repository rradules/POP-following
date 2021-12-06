import json
import matplotlib
import pandas as pd
from utils import mkdir_p, multiplicative_epsilon_metric
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set()
sns.despine()
sns.set_context(rc={"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15,
                             "legend.fontsize": 16})
sns.set_style('white', {'axes.edgecolor': "0.5", "pdf.fonttype": 42})
plt.gcf().subplots_adjust(bottom=0.15, left=0.14)

if __name__ == '__main__':

    params = {'method': 'PQL', 'novec': 30, 'states': 10, 'obj': 2, 'act': 2, \
              'suc': 4, 'seed': 42, 'exp_seed': 1, 'opt': 'ils', 'reps': 10, 'batch': 32}

    path_data = f'results/'
    path_plots = f'plots/'
    mkdir_p(path_plots)
    file = f'{params["method"]}_s{params["states"]}_a{params["act"]}_o{params["obj"]}_' \
           f'ss{params["suc"]}_seed{params["seed"]}_novec{params["novec"]}_exp{params["exp_seed"]}'

    with open(f'{path_data}ND_results_all_{file}_{params["batch"]}_reps{params["reps"]}.json', "r") as read_file:
        info = json.load(read_file)
    v0 = info['v0']

    # results = pd.read_csv(f'{path_data}comp_results_{file}_reps{params["reps"]}.csv')
    results = pd.read_csv(f'{path_data}ND_results_all_{file}_{params["batch"]}_reps{params["reps"]}.csv')
    results = results.loc[(results['Method'] == 'ils') & (results['Repetitions'] == 10)]
    perturbations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    toplot = []
    for opt_str in ['ils']:  # ['nn', 'ls', 'mls', 'ils']
        for perturb in perturbations:
            val_mean = results[['Value0', 'Value1']].loc[results['Perturbation'] == perturb].mean(axis=0).values
            val_diff = v0 - val_mean
            if params["states"] > 100:
                val_diff = multiplicative_epsilon_metric(val_mean, v0)
                toplot.append([opt_str, perturb, val_diff])
                print(opt_str, perturb, val_diff)
                print(opt_str, val_diff)
            else:
                val_diff = v0 - val_mean
                toplot.append([opt_str, perturb, max(0, max(val_diff))])
                print(opt_str, perturb, max(0, max(val_diff)))

    columns = ['Method', 'Perturbation', 'epsilon']
    df = pd.DataFrame(toplot, columns=columns)
    df.to_csv(f'{path_data}ils_pert_{file}.csv', index=False)

    ###### PLOTS #########
    results.replace('ils', 'ILS', inplace=True)
    ax = sns.barplot(x='Perturbation', y='Runtime', hue='Method', data=results, ci='sd')
    ax.set(ylabel='Average runtime (s)')
    ax.set_ylim(0, 10)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'),
                    (p.get_x() + p.get_width() / 2., p.get_height() + 1.),
                    ha='center', va='center',
                    xytext=(0, 3),
                    textcoords='offset points')
    plot_name = f"{path_plots}/{file}_ils_runtime"
    plt.savefig(plot_name + ".pdf")
    plt.clf()

    ###### PLOTS #########
    df.replace('ils', 'ILS', inplace=True)
    ax = sns.lineplot(x='Perturbation', y='epsilon', data=df, ci='sd', hue='Method')
    ax.set(ylabel='Epsilon metric')
    ax.set(xlabel='Perturbation')
    ax.set_ylim(-0.01, 1.5)

    plot_name = f"{path_plots}/pert_{file}"
    # plt.title(f"Action probabilities - Agent 2")
    plt.savefig(plot_name + ".pdf")

    plt.clf()

    # with open(f'{path_data}comp_results_all_{file}_reps_all.json', "r") as read_file:
    #    info = json.load(read_file)
    # v0 = info['v0']

    # results = pd.read_csv(f'{path_data}comp_results_all_{file}_reps_all.csv')
    results = pd.read_csv(f'{path_data}ND_results_all_{file}_{params["batch"]}_reps{params["reps"]}.csv')
    results = results.loc[((results['Method'] == 'ils') & (results['Perturbation'] == 0.3)) |
                          (results['Method'] == 'mls')]
    lsrepetitions = [5, 10, 15, 20, 25, 30, 35, 40]
    toplot2 = []
    for opt_str in ['mls', 'ils']:  # ['nn', 'ls', 'mls', 'ils']
        for reps in lsrepetitions:
            val_mean = results[['Value0', 'Value1']].loc[(results['Repetitions'] == reps)
                                                         & (results['Method'] == opt_str)].mean(axis=0).values
            val_diff = v0 - val_mean
            toplot2.append([opt_str, reps, max(0, max(val_diff))])
            print(opt_str, reps, max(0, max(val_diff)))

    columns = ['Method', 'Repetitions', 'epsilon']
    df = pd.DataFrame(toplot2, columns=columns)
    df.to_csv(f'{path_data}mls_ils_reps_{file}.csv', index=False)

    ###### PLOTS #########
    sns.set_context(rc={"font.size": 14})
    results.replace('ils', 'ILS', inplace=True)
    results.replace('mls', 'MLS', inplace=True)
    ax = sns.barplot(x='Repetitions', y='Runtime', hue='Method', data=results, ci='sd')
    ax.set(ylabel='Average runtime (s)')
    ax.set_ylim(0, 15)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'),
                    (p.get_x() + p.get_width() / 2., p.get_height() + 1.),
                    ha='center', va='center',
                    xytext=(0, 1),
                    textcoords='offset points')
    plot_name = f"{path_plots}/{file}_mls_ils_runtime"
    plt.savefig(plot_name + ".pdf")
    plt.clf()

    ls = 0.8401
    #ls = 0.2359
    #ls = 1.1849
    #ls = 0.4806

    ###### PLOTS #########
    df.replace('ils', 'ILS', inplace=True)
    df.replace('mls', 'MLS', inplace=True)
    ax = sns.lineplot(x='Repetitions', y='epsilon', data=df, ci='sd', hue='Method')
    ax.set(ylabel='Epsilon metric')
    ax.set(xlabel='Iterations')
    ax.set_ylim(-0.01, 1.5)
    ax.set_xlim(5, 40)

    ax.axhline(ls, linestyle='--', color='g', label='LS')
    plt.legend()

    plot_name = f"{path_plots}/reps_{file}"
    # plt.title(f"Action probabilities - Agent 2")
    plt.savefig(plot_name + ".pdf")

    plt.clf()

    '''
    params = {'states': 10, 'obj': 2, 'act': 2, 'suc': 4, 'seed': 42, 'exp_seed': 1, 'opt': 'nn', 'reps': 5}

    path_data = f'results/'
    path_plots = f'plots/'
    mkdir_p(path_plots)
    file = f'MPD_s{params["states"]}_a{params["act"]}_o{params["obj"]}_' \
           f'ss{params["suc"]}_seed{params["seed"]}_exp{params["exp_seed"]}'

    with open(f'{path_data}results_{params["opt"]}_{file}_reps{params["reps"]}.json', "r") as read_file:
        info = json.load(read_file)
    v0 = info['v0']
    results = []
    mtds = ['nn', 'ls', 'mls', 'ils']

    results = pd.read_csv(f'{path_data}results_all_{file}_reps{params["reps"]}.csv')

    for opt_str in mtds:
        val_mean = results[['Value0', 'Value1']].loc[results['Method'] == opt_str].mean(axis=0).values
        val_diff = v0 - val_mean
        print(opt_str, max(0, max(val_diff)))

    ###### PLOTS #########
    results.replace('nn', 'POP-NN', inplace=True)
    results.replace('ls', 'LS', inplace=True)
    results.replace('ils', 'ILS', inplace=True)
    results.replace('mls', 'MLS', inplace=True)
    ax = sns.barplot(x='Method', y='Runtime', data=results, ci='sd')
    ax.set(ylabel='Average runtime (s)')
    ax.set_ylim(0, 23)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'),
                    (p.get_x() + p.get_width() / 2. + 0.21, p.get_height() + 1),
                    ha='center', va='center',
                    xytext=(0, 3),
                    textcoords='offset points')
    plot_name = f"{path_plots}/{file}_runtime"
    # plt.title(f"Action probabilities - Agent 2")
    plt.savefig(plot_name + ".pdf")

    plt.clf()
    

    params = {'states': 10, 'obj': 2, 'act': 2, 'suc': 4, 'seed': 42, 'exp_seed': 1, 'opt': 'ils', 'reps': 10}
    momdp = 'MOMDP1'

    #params = {'states': 20, 'obj': 2, 'act': 3, 'suc': 7, 'seed': 42, 'exp_seed': 2, 'opt': 'ils', 'reps': 10}
    #momdp = 'MOMDP2'

    path_data = f'results/'
    path_plots = f'plots/'
    mkdir_p(path_plots)
    file = f'MPD_s{params["states"]}_a{params["act"]}_o{params["obj"]}_' \
           f'ss{params["suc"]}_seed{params["seed"]}_exp{params["exp_seed"]}'

    with open(f'{path_data}results_{params["opt"]}_{file}_reps{params["reps"]}.json', "r") as read_file:
        info = json.load(read_file)

    v0 = info['v0']
    results = []
    mtds = ['mls', 'ils']
    toplot = []

    reps = [5] #, 15, 20, 25]
    r10 = pd.read_csv(f'{path_data}results_all_{file}_reps{params["reps"]}.csv')
    val_ils_10 = r10[['Value0', 'Value1']].loc[r10['Method'] == 'ils'].mean(axis=0).values
    val_mls_10 = r10[['Value0', 'Value1']].loc[r10['Method'] == 'mls'].mean(axis=0).values
    val_mdiff = v0 - val_mls_10
    val_idiff = v0 - val_ils_10
    toplot.append(['mls', 10, max(0, max(val_mdiff))])
    toplot.append(['ils', 10, max(0, max(val_idiff))])
    print('mls', 10, max(0, max(val_mdiff)))
    print('ils', 10, max(0, max(val_idiff)))

    for opt in mtds:
        for r in reps:
            df = pd.read_csv(f'{path_data}results_{opt}_{file}_reps{r}.csv')
            df['reps'] = r
            results.append(df)

    results = pd.concat(results, axis=0, sort=False)

    for opt_str in mtds:
        for r in reps:
            val_mean = results[['Value0', 'Value1']].loc[(results['Method'] == opt_str) & (results['reps'] == r)].mean(axis=0).values
            val_diff = v0 - val_mean
            toplot.append([opt_str, r, max(0, max(val_diff))])
            print(opt_str, r, max(0, max(val_diff)))

    for opt_str in mtds:
        for r in [15, 20, 25]:
            toplot.append([opt_str, r, 0])

    columns = ['Method', 'Repetitions', 'epsilon']
    df = pd.DataFrame(toplot, columns=columns)
    df.to_csv(f'{path_data}results_{momdp}_ils_mls.csv', index=False)

    df = pd.read_csv(f'{path_data}results_{momdp}_ils_mls.csv')

    ###### PLOTS #########
    df.replace('ils', 'ILS', inplace=True)
    df.replace('mls', 'MLS', inplace=True)
    ax = sns.lineplot(x='Repetitions', y='epsilon', data=df, ci='sd', hue='Method')
    ax.set(ylabel='Epsilon metric')
    ax.set(xlabel='Iterations')
    ax.set_ylim(-0.01, 0.12)

    plot_name = f"{path_plots}/{momdp}_reps"
    # plt.title(f"Action probabilities - Agent 2")
    plt.savefig(plot_name + ".pdf")

    plt.clf()



    '''
