import torch
import networkx as nx
import numpy as np
import pandas as pd

from scipy.stats import entropy
from statistics import mean, stdev

from eden.graph import vectorize
from utils.constants import ORDER_DIR, RUNS_DIR
from utils.evaluation import nspdk
from utils.serializer import load_yaml, save_yaml

ORDERS = ["bfs-fixed", "dfs-fixed", "bfs-random", "dfs-random", "random", "smiles"]
MODELS = ["BA", "ER", "GRU", "GRAPHRNN", "GRAPHER"]
DATASETS = ["ladders", "community", "ego", "ENZYMES", "PROTEINS_full", "trees"]
QUAL_METRICS = ["betweenness", "clustering", "degree", "orbit", "nspdk"]
QUANT_METRICS = ["novelty1000", "novelty5000", "uniqueness1000", "uniqueness5000"]
ALL_METRICS = QUAL_METRICS + QUANT_METRICS


def patch_graphs(graphlist):
    graphs = []
    for i, _ in enumerate(graphlist):
        G = graphlist[i]
        G = nx.convert_node_labels_to_integers(G)
        for node in G.nodes():
            G.nodes[node]["label"] = str(nx.degree(G, node))
        graphs.append(G)
    return graphs


def load_result(root, model, dataset):
    path = root / model / dataset / "results" / f"{dataset}.yaml"
    return load_yaml(path)


def load_qual_samples(root, model, dataset):
    path = root / model / dataset / "samples"
    samples = []
    for filepath in sorted(list(path.iterdir())):
        if '000' in filepath.stem:
            continue
        graphlist = torch.load(filepath)
        samples.append(patch_graphs(graphlist))
    return samples


def load_test_set(root, model, dataset):
    path = root / model / dataset / "data"
    splits = load_yaml(path / "splits.yaml")
    dataset = torch.load(path / f"{dataset}.pt")
    graphs = patch_graphs(dataset.graphlist)
    graphs = [graphs[i] for i in splits["test"]]
    return graphs


def calculate_nspdk(model, dataset):
    samples = load_qual_samples(model, dataset)
    ref = load_test_set(model, dataset)
    values = [float(nspdk(ref, sample)) for sample in samples]
    return float(mean(values)), float(stdev(values)), values


def collate_experiments():
    all_data = []

    for model in MODELS:
        for dataset in DATASETS:
            result = load_result(RUNS_DIR, model, dataset)

            for metric in QUAL_METRICS:
                m, s = result[metric]["mean"], result[metric]["std"]
                row = {"model": model, "dataset": dataset, "metric": metric, "avg": m, "stdev": s}
                all_data.append(row)

            for metric in QUANT_METRICS:
                m = result[metric]
                row = {"model": model, "dataset": dataset, "metric": metric, "avg": m, "stdev": None}
                all_data.append(row)

    return pd.DataFrame(all_data)


def collate_order_experiments():
    all_data = []

    for order in ORDERS[1:]:
        for dataset in ["trees"]:
            if order == "smiles" and dataset not in ["PROTEINS_full", "ENZYMES"]:
                continue

            result = load_result(ORDER_DIR, order, dataset)

            if 'nspdk' not in result:
                ref = load_test_set(ORDER_DIR, order, dataset)
                samples = load_qual_samples(ORDER_DIR, order, dataset)
                m, s, v = nspdk(ref, samples)
                result.update(nspdk={"mean": m, "std": s, "scores": v})
                save_yaml(result, ORDER_DIR / order/ dataset / "results" / f"{dataset}.yaml")

            for metric in QUAL_METRICS:
                m, s = result[metric]["mean"], result[metric]["std"]
                row = {"order": order, "dataset": dataset, "metric": metric, "avg": m, "stdev": s}
                all_data.append(row)

    return pd.DataFrame(all_data)