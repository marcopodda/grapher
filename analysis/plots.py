import torch
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

from analysis.collect import parse_log, collate_results, QUAL_METRICS, DATASETS
from utils.constants import HUMANIZE


np.random.seed(42)


plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


DRAW_KWARGS = {
    "with_labels": False,
    "node_size": 0.5,
    "width": 0.3
}


def plot_kde(save_path="../phd-thesis/Figures/Chapter6/displot.eps"):
    plt.rcParams.update({ 'font.size': 20})

    models = ["Data", "Ours", "GraphRNN"]
    metrics = [HUMANIZE[m] for m in QUAL_METRICS[:-1]]
    datasets = [HUMANIZE[d] for d in DATASETS]

    data = collate_results()
    data = data[data.Model.isin(models)]
    # data = data[data.Value>0]

    g = sns.displot(
        x="Value", hue="Model", row="Dataset", col="Metric", kind="kde",
        data=data, common_norm=False, hue_order=models, row_order=datasets, col_order=metrics,
        facet_kws=dict(sharex=False, sharey=False), height=3)

    for i, _ in enumerate(datasets):
        for j, _ in enumerate(metrics):
            g.axes[i,j].set_xlabel("")
            g.axes[i,j].set_ylabel("")

    for i, metric in enumerate(metrics):
        g.axes[-1,i].set_xlabel(metric)

    for i, dataset in enumerate(datasets):
        g.axes[i,0].set_ylabel(dataset)

    g.set_titles(template="", col_template="", row_template="")
    g.set_xticklabels(labels=[])
    g.set_yticklabels(labels=[])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


def plot_loss(log_path, save_path="../phd-thesis/Figures/Chapter6/loss.eps"):
    log = parse_log(log_path)
    hue_order = ["BFS", "DFS", "RANDOM", "BFS RANDOM", "DFS RANDOM", "SMILES"]
    g = sns.FacetGrid(log, col="Dataset", hue="Order", col_wrap=2, hue_order=hue_order, height=3, sharex=False)
    g.map(sns.lineplot, "Epoch", "Loss")
    g.add_legend()
    g.set_titles(col_template="{col_name}", row_template="{row_name}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


def draw_ladder_graph(G, ax):
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, ax=ax, **DRAW_KWARGS)


def draw_standard_graph(G, ax):
    nx.draw_networkx(G, pos=None, ax=ax, **DRAW_KWARGS)


def draw_tree_graph(G, ax):
    pos = graphviz_layout(G, prog="twopi")
    nx.draw_networkx(G, pos=pos, ax=ax, **DRAW_KWARGS)


def draw_molecular_graph(G, ax):
    draw_standard_graph(G, ax)


DRAW_FUNCS = {
    "ladders": draw_ladder_graph,
    "community": draw_standard_graph,
    "ego": draw_standard_graph,
    "trees": draw_tree_graph,
    "ENZYMES": draw_molecular_graph,
    "PROTEINS_full": draw_molecular_graph
}


def plot_samples(graphs_per_dataset=5, save_path="../phd-thesis/Figures/Chapter6/samples.eps"):
    num_datasets = len(DATASETS)
    col_offset = graphs_per_dataset + 1

    def random_sample(graphs, num_samples):
        indices = np.random.choice(len(graphs), num_samples, replace=False)
        return [graphs[i] for i in indices]

    def load_generated_samples(dataset, num_samples):
        samples = torch.load(f"RUNS/GRAPHER/{dataset}/samples/samples.pt")
        return random_sample(samples, num_samples)

    def load_real_samples(dataset, num_samples):
        samples = torch.load(f"RUNS/GRAPHER/{dataset}/data/{dataset}.pt").graphlist
        return random_sample(samples, num_samples)

    def clean_ax(ax):
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        return ax

    _, axs = plt.subplots(num_datasets, graphs_per_dataset * 2 + 1, figsize=(6, 6))

    for i, dataset in enumerate(DATASETS):
        real_samples = load_real_samples(dataset, num_samples=graphs_per_dataset)
        generated_samples = load_generated_samples(dataset, num_samples=graphs_per_dataset)
        draw_function = DRAW_FUNCS[dataset]

        for j in range(graphs_per_dataset):
            G = real_samples[j]
            ax = clean_ax(axs[i][j])
            draw_function(G, ax)

        clean_ax(axs[i][graphs_per_dataset])

        for j in range(graphs_per_dataset):
            ax = clean_ax(axs[i][j + col_offset])
            G = generated_samples[j]
            draw_function(G, ax)

        axs[i][0].set_ylabel(HUMANIZE[dataset])

    pos_center = graphs_per_dataset // 2
    axs[0][pos_center].set_title("REAL")
    axs[0][pos_center + graphs_per_dataset + 1].set_title("GENERATED")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)