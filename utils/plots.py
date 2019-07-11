import torch
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import seaborn as sns

from pathlib import Path
from learner.trainer import Trainer
from config.config import Config
from dataset import get_dataset_class
from .constants import ORDER_NAMES, MODEL_NAMES, DATASET_NAMES
from .serializer import load_yaml

sns.set('paper')
sns.set_style('whitegrid')


HUMANIZE_DATASET = {
    "community": "Community",
    "PROTEINS_full": "Protein",
    "ladders": "Ladders",
    "ENZYMES": "Enzymes",
    "ego": "Ego"
}

HUMANIZE_ORDER = {
    "random": "Random",
    "bfs-random": "BF Random",
    "smiles": "SMILES",
}

HUMANIZE_METRIC = {
    "degree": "Average Degree Distribution",
    "clustering": "Average Clustering Coefficient",
    "orbit": "Average Orbit Count"
}


root = Path('RUNS')
order_root = Path('RUNS') / "ORDER"


def _to_hist_data(v):
    data = []
    for x in v:
        data.extend([v.index(x)] * int(np.round(x)))
    return data


def _double_plot(ref, pred, metric_name, model_name, dataset_name):
    data1 = _to_hist_data(pred)
    data2 = _to_hist_data(ref)
    sns.distplot(data1, hist=False, label="Generated")
    ax = sns.distplot(data2, hist=False, label="Real")
    ax.set_title(HUMANIZE_METRIC[metric_name])
    if metric_name == "degree":
        ax.set_ylabel(HUMANIZE_DATASET[dataset_name])
    ax.legend()
    fig = plt.gcf()
    fig.savefig(f"{model_name}_{dataset_name}_{metric_name}.svg")


def load_result(model_name, dataset_name, order):
    path = root if order is False else order_root
    path = path / model_name / dataset_name
    result_path = list(path.glob("*"))[-1] / "results" / f"{dataset_name}.yaml"
    return load_yaml(result_path)


def plot_stat(metric_name, model_name, dataset_name, order=False):
    result = load_result(model_name, dataset_name, order=order)[metric_name]
    ref = result['data_hist']
    pred = result['samples_hist']
    _double_plot(ref, pred, metric_name, model_name, dataset_name)


def plot_training_on_ordering(dataset_name):
    path = Path("RUNS") / "GRAPHER"
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

    path = Path("RUNS") / "ORDER"
    for order in reversed(ORDER_NAMES):
        try:
            exp_path = list((path / order / dataset_name).glob("*"))[0]
        except IndexError:
            continue
        config = Config.from_file(exp_path/"config"/"config.yaml")
        dataset_class = get_dataset_class(dataset_name)
        dataset = dataset_class(config, exp_path, dataset_name)
        trainer = Trainer.load(config, exp_path, dataset.input_dim, dataset.output_dim, best=True)
        loss = np.array(trainer.losses1) + np.array(trainer.losses2)
        plt.plot(loss, label=HUMANIZE_ORDER[order])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_generated_graph(model_name, dataset_name, layout="spring"):
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
