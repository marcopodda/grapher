from os import replace
import torch
import networkx as nx
import numpy as np
import pandas as pd

from statistics import mean, stdev

from utils.constants import DATASET_NAMES, ORDER_DIR, RUNS_DIR, ROOT, HUMANIZE
from utils.evaluation import nspdk
from utils.serializer import load_yaml, save_yaml

ORDERS = ["bfs-fixed", "random", "dfs-random", "bfs-random", "dfs-fixed", "smiles"]
MODELS = ["BA", "ER", "GRU", "GRAPHRNN", "GRAPHER"]
DATASETS = ["ladders", "community", "ego", "trees", "ENZYMES", "PROTEINS_full"]
QUAL_METRICS = ["degree", "clustering", "orbit", "betweenness", "nspdk"]
QUANT_METRICS = ["novelty1000", "novelty5000", "uniqueness1000", "uniqueness5000"]


def collate_metric(metric_data):
    num_trials = len(metric_data)
    refs, gens, scores = [], [], []

    for num_trial in range(num_trials):
        elem = metric_data[num_trial]
        refs.append(elem["ref"])
        gens.append(elem["gen"])
        scores.append(elem["score"])

    return {
        "avg": np.mean(scores),
        "std": np.std(scores),
        "ref": np.array(refs).mean(axis=0).round(),
        "gen": np.array(gens).mean(axis=0).round()
    }


def compute_mean(res, metric):
    values = []
    for i in range(len(res[metric])):
        values.append(res[metric][i]["score"])
    return np.mean(values), np.std(values)


def collate_results():
    rows = []

    for dataset in DATASET_NAMES:
        for model in MODELS:
            path = RUNS_DIR / model / dataset / "results" / "results.pt"
            if path.exists():
                result = torch.load(path)
                for metric in QUAL_METRICS[:-1]:
                    metric_data = collate_metric(result[metric])
                    ref_data = metric_data["ref"]
                    for i in range(ref_data.shape[0]):
                        rows.append({
                            "Model": "Data",
                            "Dataset": HUMANIZE[dataset],
                            "Metric": HUMANIZE[metric],
                            "Value": ref_data[i]
                        })
                    gen_data = metric_data["gen"]
                    for i in range(gen_data.shape[0]):
                        rows.append({
                            "Model": HUMANIZE[model],
                            "Dataset": HUMANIZE[dataset],
                            "Metric": HUMANIZE[metric],
                            "Value": gen_data[i]
                        })

    return pd.DataFrame(rows)


def collate_dataset_result(dataset):
    rows = []
    for model in MODELS:
        result = torch.load(f"RUNS/{model}/{dataset}/results/results.pt")
        for metric in QUAL_METRICS:
            mean, std = compute_mean(result, metric)
            value = f"{mean:.3f} ({std:.3f})"
            rows.append(dict(dataset=dataset, metric=metric, value=value))
    return pd.DataFrame(rows)


def collate_dataset_row(dataset, metric):
    df = collate_dataset_result(dataset)
    df = df[df.metric==metric].T
    row = df.loc["value",:]
    score = row.values.tolist()
    print(" & ".join(score) + "\\\\")


def collate_order_dataset_result(dataset):
    rows = []
    for order in ORDERS[1:]:
        if order == "smiles" and dataset not in ["PROTEINS_full", "ENZYMES"]:
            continue
        result = torch.load(f"RUNS/ORDER/{order}/{dataset}/results/results.pt")
        for metric in QUAL_METRICS:
            mean, std = compute_mean(result, metric)
            value = f"{mean:.3f} ({std:.3f})"
            rows.append(dict(dataset=dataset, metric=metric, value=value))
    return pd.DataFrame(rows)


def collate_order_row(dataset, metric):
    df = collate_order_dataset_result(dataset)
    df = df[df.metric==metric].T
    row = df.loc["value",:]
    score = row.values.tolist()
    print(" & ".join(score) + "\\\\")


def parse_log(path):
    rows = []
    dataset, order = None, None
    with open(path, "r") as f:
        for l in f.readlines():
            if l.startswith("Training"):
                line = l.split(" ")
                dataset, order = line[2], line[5].rsplit("\n")[0]
            else:
                epoch, _, _, lt, _ = l.split(" - ")
                _, value = lt.split(": ")
                rows.append({
                    "Dataset": HUMANIZE[dataset],
                    "Order": HUMANIZE[order],
                    "Loss": float(value),
                    "Epoch": int(epoch)
                })

    return pd.DataFrame(rows)
