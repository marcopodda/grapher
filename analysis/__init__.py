import torch
import networkx as nx
import numpy as np
import pandas as pd

from scipy.stats import entropy
from statistics import mean, stdev

from eden.graph import vectorize
from utils.constants import RUNS_DIR
from utils.evaluation import nspdk
from utils.serializer import load_yaml, save_yaml


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


def load_result(model, dataset):
    path = RUNS_DIR / model / dataset / "results" / f"{dataset}.yaml"
    return load_yaml(path)


def load_qual_samples(model, dataset):
    path = RUNS_DIR / model / dataset / "samples"
    samples = []
    for filepath in sorted(list(path.iterdir())):
        if '000' in filepath.stem:
            continue
        graphlist = torch.load(filepath)
        samples.append(patch_graphs(graphlist))
    return samples


def load_test_set(model, dataset):
    path = RUNS_DIR / model / dataset / "data"
    splits = load_yaml(path / "splits.yaml")
    dataset = torch.load(path / f"{dataset}.pt")
    graphs = patch_graphs(dataset.graphlist)
    graphs = [graphs[i] for i in splits["test"]]
    return graphs


def collate_experiments():
    all_data = []

    for model in MODELS:
        for dataset in DATASETS:
            result = load_result(model, dataset)

            if not 'nspdk' in result:
                m, s, v = calculate_nspdk(model, dataset)
                result['nspdk'] = {"mean": m, "std": s, "scores": v}
                save_yaml(result, RUNS_DIR / model / dataset / "results" / f"{dataset}.yaml")

            for metric in QUAL_METRICS:
                m, s = result[metric]["mean"], result[metric]["std"]
                row = {"model": model, "dataset": dataset, "metric": metric, "avg": m, "stdev": s}
                all_data.append(row)

            for metric in QUANT_METRICS:
                m = result[metric]
                row = {"model": model, "dataset": dataset, "metric": metric, "avg": m, "stdev": None}
                all_data.append(row)

    return pd.DataFrame(all_data)


def calculate_nspdk(model, dataset):
    samples = load_qual_samples(model, dataset)
    ref = load_test_set(model, dataset)
    values = [nspdk(ref, sample) for sample in samples]
    return mean(values), stdev(values), values
