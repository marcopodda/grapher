import torch
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import seaborn as sns

from pathlib import Path
from learner.trainer import Trainer
from config.config import Config
from dataset import get_dataset_class


sns.set('paper')
sns.set_style('whitegrid')


HUMANIZE_DATASET = {
    "community": "Community",
    "PROTEINS_full": "Protein",
    "ladders": "Ladders",
    "ENZYMES": "Enzymes",
    "ego": "Ego"
}


def _to_hist_data(v):
    data = []
    for x in v:
        data.extend([v.index(x)] * int(np.round(x)))
    return data


def _double_plot(ref, pred, stat_name, model_name, dataset_name):
    data1 = _to_hist_data(ref)
    data2 = _to_hist_data(pred)
    sns.distplot(data1, hist=False, label="Model")
    ax = sns.distplot(data2, hist=False, label="Data")
    ax.legend()
    fig = plt.gcf()
    fig.savefig(f"{model_name}_{dataset_name}_{stat_name}.svg")


def plot_stat(results, stat_name, model_name, dataset_name):
    ref_name = f"{stat_name}_count_data"
    pred_name = f"{stat_name}_count_samples"
    ref = results[model_name][dataset_name][ref_name]
    pred = results[model_name][dataset_name][pred_name]
    _double_plot(ref, pred, stat_name, model_name, dataset_name)


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
    for order in ["smiles", "bfs-random", "random"]:
        try:
            exp_path = list((path / order / dataset_name).glob("*"))[0]
        except IndexError:
            continue
        config = Config.from_file(exp_path/"config"/"config.yaml")
        dataset_class = get_dataset_class(dataset_name)
        dataset = dataset_class(config, exp_path, dataset_name)
        trainer = Trainer.load(config, exp_path, dataset.input_dim, dataset.output_dim, best=True)
        loss = np.array(trainer.losses1) + np.array(trainer.losses2)
        plt.plot(loss, label=order)
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
    pred = torch.load(exp_path / "samples" / "samples_0.pt")

    G_ref = np.random.choice(ref, size=1)[0]
    mapping = {n: i for (i, n) in enumerate(G_ref.nodes())}
    G_ref = nx.relabel_nodes(G_ref, mapping)

    G_pred = np.random.choice(pred, size=1)[0]
    mapping = {n: i for (i, n) in enumerate(G_pred.nodes())}
    G_pred = nx.relabel_nodes(G_pred, mapping)

    fig, axs = plt.subplots(1, 2)
    layout_fn = getattr(nx, f"{layout}_layout")
    nx.draw_networkx(G_ref, ax=axs[0], pos=layout_fn(G_ref), with_labels=False, node_size=10)
    nx.draw_networkx(G_pred, ax=axs[1], pos=layout_fn(G_pred), with_labels=False, node_size=10)

    axs[0].set_ylabel(HUMANIZE_DATASET[dataset_name])
    axs[0].set_title("Real")
    axs[0].grid(False)
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].set_title("Generated")
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].grid(False)

    plt.show()
