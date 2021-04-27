from os import replace
import torch
import networkx as nx
import numpy as np
import pandas as pd

from statistics import mean, stdev

from utils.constants import DATASET_NAMES, ORDER_DIR, RUNS_DIR, ROOT, HUMANIZE
from utils.evaluation import nspdk
from utils.serializer import load_yaml, save_yaml

ORDERS = ["bfs-fixed", "dfs-fixed", "bfs-random", "dfs-random", "random", "smiles"]
MODELS = ["BA", "ER", "GRU", "GRAPHRNN", "GRAPHER"]
DATASETS = ["ladders", "community", "ego", "ENZYMES", "PROTEINS_full", "trees"]
QUAL_METRICS = ["betweenness", "clustering", "degree", "orbit", "nspdk"]
QUANT_METRICS = ["novelty1000", "novelty5000", "uniqueness1000", "uniqueness5000"]
ALL_METRICS = QUAL_METRICS + QUANT_METRICS


def load_result(root, model, dataset):
    path = root / model / dataset / "results" / f"results.pt"
    return torch.load(path)


def collate_experiments():
    all_data = []

    for model in MODELS:
        for dataset in DATASETS:
            result = load_result(RUNS_DIR, model, dataset)

            for metric in QUAL_METRICS:
                m, s = result[metric]["mean"], result[metric]["std"]
                row = {"Model": model, "Dataset": HUMANIZE[dataset], "Metric": HUMANIZE[metric], "avg": m, "stdev": s}
                all_data.append(row)

            for metric in QUANT_METRICS:
                m = result[metric]
                row = {"Model": model, "Dataset": HUMANIZE[dataset], "Metric": HUMANIZE[metric], "avg": m, "stdev": None}
                all_data.append(row)

    all_data = pd.DataFrame(all_data)
    return all_data


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
        "ref": np.array(refs).mean(axis=0),
        "gen": np.array(gens).mean(axis=0)
    }



def collate_results():
    rows = []

    for dataset in DATASET_NAMES:
        for model in MODELS:
            path = RUNS_DIR / model / dataset / "results" / "results.pt"
            if path.exists():
                result = torch.load(path)
                for metric in QUAL_METRICS:
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
                            "Model": model,
                            "Dataset": HUMANIZE[dataset],
                            "Metric": HUMANIZE[metric],
                            "Value": gen_data[i]
                        })

    return pd.DataFrame(rows)



def collate_order_experiments():
    all_data = []

    for order in ORDERS:
        for dataset in ["trees"]:
            if order == "smiles" and dataset not in ["PROTEINS_full", "ENZYMES"]:
                continue

            if order == "bfs-fixed":
                result = load_result(RUNS_DIR, "GRAPHER", dataset)
            else:
                result = load_result(ORDER_DIR, order, dataset)

            for metric in QUAL_METRICS:
                m, s = result[metric]["mean"], result[metric]["std"]
                row = {"Order": HUMANIZE[order], "Dataset": HUMANIZE[dataset], "Metric": HUMANIZE[metric], "avg": m, "stdev": s}
                all_data.append(row)

    return pd.DataFrame(all_data)


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
