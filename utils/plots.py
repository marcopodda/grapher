import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


sns.set('paper')
sns.set_style('whitegrid')


def _to_hist_data(v):
    data = []
    for x in v:
        data.extend([v.index(x)] * int(np.round(x)))
    return data


def _double_plot(ref, pred, stat_name, model_name, dataset_name):
    data1 = _to_hist_data(ref)
    data2 = _to_hist_data(pred)
    print(len(data1), len(data2))
    sns.distplot(data1, hist=False, label="Model")
    ax = sns.distplot(data2, hist=False, label="Data")
    ax.legend()
    fname = f"{stat_name}_{model_name}_{dataset_name}.svg"
    fig = plt.gcf()
    fig.savefig(fname)


def plot_stat(results, stat_name, model_name, dataset_name):
    ref_name = f"{stat_name}_count_data"
    pred_name = f"{stat_name}_count_samples"
    ref = results[model_name][dataset_name][ref_name]
    pred = results[model_name][dataset_name][pred_name]
    _double_plot(ref, pred, stat_name, model_name, dataset_name)


def plot_training():
    pass
