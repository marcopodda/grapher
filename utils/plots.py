import torch
import numpy as np
import networkx as nx
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

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


def _get_legend_handles():
    labels = ["Real", "Generated"]
    colors = sns.color_palette().as_hex()
    label2color = {m: c for (m,c) in zip(labels, colors)}
    handles = [Line2D([0],[0], color=label2color[l], lw=3, label=l) for l in labels]
    return handles


def _plot_stat(metric_name, model_name, dataset_name, order=False, ax=None):
    def to_histogram(v):
        data = []
        for x in v:
            data.extend([v.index(x)] * int(np.round(x)))
        return data

    result = load_result(model_name, dataset_name, order=order)[metric_name]
    ref_histo = to_histogram(result['data_hist'])
    samples_histo = to_histogram(result['samples_hist'])
    sns.distplot(ref_histo, hist=False, ax=ax)
    sns.distplot(samples_histo, hist=False, ax=ax)



def plot_real_vs_samples_distributions(model_name="GRAPHER", dataset_names=["community", "ENZYMES"]):
    if not isinstance(dataset_names, list):
        dataset_names = list(dataset_names)

    nrows, ncols = len(dataset_names), len(QUALITATIVE_METRIC_NAMES)
    legend = None
    fig, axs = plt.subplots(nrows, ncols)

    for row, dataset_name in enumerate(dataset_names):
        for col, metric_name in enumerate(QUALITATIVE_METRIC_NAMES):
            ax = axs[row][col]
            ax.set_xticks([])
            ax.set_yticks([])
            _plot_stat(metric_name, model_name, dataset_name, order=False, ax=ax)

    for row in range(nrows):
        name = HUMANIZE_DATASET[dataset_names[row]]
        axs[row][0].set_ylabel(name)

    for col in range(ncols):
        name = HUMANIZE_METRIC[QUALITATIVE_METRIC_NAMES[col]]
        name = "\n".join(name.split(" "))
        axs[0][col].set_title(name)

    legend_handles = _get_legend_handles()
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=9)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06)

    fig.set_size_inches(9, 9)
    plt.savefig("plot.eps", dpi=300)


def plot_orderings():
    fig, axs = plt.subplots(1, 3)
    for i, dataset_name in enumerate(["community", "ego", "PROTEINS_full"]):
        plot_loss_with_different_orderings(axs[i], dataset_name)
    axs[0].set_ylabel("Loss")
    
    fig.savefig("fig6.eps", dpi=300)


def plot_loss_with_different_orderings(ax, dataset_name):
    path = RUNS_DIR / "GRAPHER"
    try:
        exp_path = list((path / dataset_name).glob("*"))[0]
    except IndexError as e:
        return

    orders = ["bfs-fixed", "dfs-fixed", "dfs-random", "bfs-random", "random", "smiles"]
    colors = sns.color_palette().as_hex()
    order2color = {o: c for (o, c) in zip(orders, colors)}

    config = Config.from_file(exp_path/"config"/"config.yaml")
    dataset_class = get_dataset_class(dataset_name)
    dataset = dataset_class(config, exp_path, dataset_name)
    trainer = Trainer.load(config, exp_path, dataset.input_dim, dataset.output_dim, best=True)
    loss = np.array(trainer.losses1) + np.array(trainer.losses2)

    ax.plot(loss, label="Ours", c=order2color["bfs-fixed"])

    for order in orders:
        path = RUNS_DIR / "ORDER"
        try:
            exp_path = list((path / order / dataset_name).glob("*"))[0]
            experiment = load_experiment(path, order, dataset_name)
        except IndexError as e:
            continue
        config = Config.from_file(exp_path/"config"/"config.yaml")
        dataset_class = get_dataset_class(dataset_name)
        dataset = dataset_class(config, exp_path, dataset_name)
        trainer = Trainer.load(config, exp_path, dataset.input_dim, dataset.output_dim, best=True)
        loss = np.array(trainer.losses1) + np.array(trainer.losses2)
        ax.plot(loss, label=HUMANIZE_ORDER[order], c=order2color[order])

    ax.legend()
    ax.set_xlabel("Epochs")
    ax.set_title(HUMANIZE_DATASET[dataset_name])


def plot_real_vs_generated_graphs(model_name, dataset_name, layout="spring"):
    
    num_graphs = 5
    datasets = ["ladders", "ego", "community", "ENZYMES", "PROTEINS_full"]

    fig, axs = plt.subplots(len(datasets), num_graphs * 2)
    for d, dataset_name in enumerate(datasets):
        path = Path("RUNS") / model_name / dataset_name

        try:
            exp_path = list((path).glob("*"))[0]
        except IndexError:
            return

        ref = torch.load(exp_path / "data" / f"{dataset_name}.pt").graphlist
        samples_path = list((exp_path / "samples").glob("*.pt"))[0]
        pred = torch.load(samples_path)

        real = np.random.choice(ref, size=num_graphs)
        gen = np.random.choice(pred, size=num_graphs)

        
        layout_fn = getattr(nx, f"{layout}_layout")
        node_size = 0.5
        line_width = 0.3
        
        for i in range(num_graphs):
            nx.draw_networkx(real[i], ax=axs[d][i], pos=layout_fn(real[i]), with_labels=False, node_size=node_size, width=line_width)
            axs[d][i].grid(False)
            axs[d][i].set_xticks([])
            axs[d][i].set_yticks([])
            axs[d][i].spines['right'].set_visible(False)
            axs[d][i].spines['top'].set_visible(False)
            axs[d][i].spines['bottom'].set_visible(False)
            axs[d][i].spines['left'].set_visible(False)
        axs[d][0].set_ylabel(HUMANIZE_DATASET[dataset_name])
        
        for i in range(num_graphs):
            nx.draw_networkx(gen[i], ax=axs[d][i+5], pos=layout_fn(gen[i]), with_labels=False, node_size=node_size, width=line_width)
            axs[d][i+5].grid(False)
            axs[d][i+5].set_xticks([])
            axs[d][i+5].set_yticks([])
            axs[d][i+5].spines['right'].set_visible(False)
            axs[d][i+5].spines['top'].set_visible(False)
            axs[d][i+5].spines['bottom'].set_visible(False)
            axs[d][i+5].spines['left'].set_visible(False)

    axs[0][2].set_title("Real")
    axs[0][7].set_title("Generated")
    fig.savefig("gen.svg", dpi=300)
    

    # axs.flat[0].set_title(HUMANIZE_DATASET[dataset_name])

    # axs.flat[0].set_ylabel("Real")
    # axs.flat[0].grid(False)
    # axs.flat[0].set_xticks([])
    # axs.flat[0].set_yticks([])

    # axs.flat[1].set_ylabel("Generated")
    # axs.flat[1].set_xticks([])
    # axs.flat[1].set_yticks([])
    # axs.flat[1].grid(False)

    fig = plt.gcf()
    # fig.set_size_inches(3, 18)

    plt.show()
