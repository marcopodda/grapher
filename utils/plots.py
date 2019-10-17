import torch
import numpy as np
import networkx as nx
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

from pathlib import Path
from learner.trainer import Trainer
from config.config import Config
from dataset import get_dataset_class

from experiment import load_experiment
from .constants import *
from .serializer import load_yaml
from .misc import load_result, last_in_folder

sns.set('paper')
sns.set_style("whitegrid", {'axes.grid' : False})


root = Path('RUNS')
order_root = Path('RUNS') / "ORDER"



def _plot_stat(metric_name, model_name, dataset_name, order=False, ax=None):
    def to_histogram(v):
        data = []
        for x in v:
            data.extend([v.index(x)] * int(np.round(x)))
        return data

    result = load_result(model_name, dataset_name, order=order)[metric_name]
    ref_histo = to_histogram(result['data_hist'])
    samples_histo = to_histogram(result['samples_hist'])
    sns.distplot(ref_histo, hist=False, label="Real", ax=ax)
    sns.distplot(samples_histo, hist=False, label="Generated", ax=ax)



def plot_real_vs_samples_distributions(model_name="GRAPHER", dataset_names=["community", "ENZYMES"]):
    if not isinstance(dataset_names, list):
        dataset_names = list(dataset_names)

    nrows, ncols = len(QUALITATIVE_METRIC_NAMES), len(dataset_names),
    legend = None
    fig, axs = plt.subplots(nrows, ncols)

    for row, metric_name in enumerate(QUALITATIVE_METRIC_NAMES):
        for col, dataset_name in enumerate(dataset_names):
            ax = axs[row][col]
            ax.set_xticks([])
            ax.set_yticks([])
            if ax.get_legend():
                ax.get_legend().remove()
            _plot_stat(metric_name, model_name, dataset_name, order=False, ax=ax)

    for col in range(ncols):
        name = HUMANIZE_DATASET[dataset_names[col]]
        axs[0][col].set_title(name)

    for row in range(nrows):
        name = HUMANIZE_METRIC[QUALITATIVE_METRIC_NAMES[row]]
        name = "\n".join(name.split(" "))
        axs[row][0].set_ylabel(name)

    fig.set_size_inches(9, 9)
    plt.show()


def plot_loss_with_different_orderings(dataset_name):
    path = RUNS_DIR / "GRAPHER"
    try:
        exp_path = list((path / dataset_name).glob("*"))[0]
    except IndexError:
        return
    config = Config.from_file(exp_path/"config"/"config.yaml")
    dataset_class = get_dataset_class(dataset_name)
    dataset = dataset_class(config, exp_path, dataset_name)
    trainer = Trainer.load(config, exp_path, dataset.input_dim, dataset.output_dim, best=True)
    loss = np.array(trainer.losses1) + np.array(trainer.losses2)

    plt.plot(loss, label="Default")

    for order in reversed(ORDER_NAMES):
        experiment = load_experiment(ORDER_DIR, order, dataset_name)
        config = Config.from_file(exp_path/"config"/"config.yaml")
        dataset_class = get_dataset_class(dataset_name)
        dataset = dataset_class(config, exp_path, dataset_name)
        trainer = Trainer.load(config, exp_path, dataset.input_dim, dataset.output_dim, best=True)
        loss = np.array(trainer.losses1) + np.array(trainer.losses2)
        plt.plot(loss, label=HUMANIZE_ORDER[order])

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.grid('off')
    plt.legend()
    plt.show()


def plot_real_vs_generated_graphs(model_name, dataset_name, layout="spring"):
    path = Path("RUNS") / model_name / dataset_name

    try:
        exp_path = list((path).glob("*"))[0]
    except IndexError:
        return

    ref = torch.load(exp_path / "data" / f"{dataset_name}.pt").graphlist
    samples_path = list((exp_path / "samples").glob("*.pt"))[0]
    pred = torch.load(samples_path)

    G_ref = np.random.choice(ref, size=1)[0]
    mapping = {n: i for (i, n) in enumerate(G_ref.nodes())}
    G_ref = nx.relabel_nodes(G_ref, mapping)

    G_pred = np.random.choice(pred, size=1)[0]
    mapping = {n: i for (i, n) in enumerate(G_pred.nodes())}
    G_pred = nx.relabel_nodes(G_pred, mapping)

    fig, axs = plt.subplots(2, 1)
    layout_fn = getattr(nx, f"{layout}_layout")
    nx.draw_networkx(G_ref, ax=axs[0], pos=layout_fn(G_ref), with_labels=False, node_size=10)
    nx.draw_networkx(G_pred, ax=axs[1], pos=layout_fn(G_pred), with_labels=False, node_size=10)

    axs[0].set_title(HUMANIZE_DATASET[dataset_name])

    axs[0].set_ylabel("Real")
    axs[0].grid(False)
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].set_ylabel("Generated")
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].grid(False)

    fig = plt.gcf()
    fig.set_size_inches(3, 18)

    plt.show()
