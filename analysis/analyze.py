import torch
import networkx as nx
import numpy as np
import pandas as pd

from statistics import mean, stdev

from utils.constants import DATASET_NAMES, MODEL_NAMES, ORDER_DIR, RUNS_DIR, ROOT
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


def calculate_nspdk(root, model, dataset):
    samples = load_qual_samples(root, model, dataset)
    ref = load_test_set(root, model, dataset)
    values = [float(nspdk(ref, sample)) for sample in samples]
    return float(mean(values)), float(stdev(values)), values


def to_histogram(v):
    data = []
    for x in v:
        data.extend([v.index(x)] * int(np.round(x)))
    return data


def collate_experiments():
    all_data, all_hist_data = [], []

    for model in MODELS:
        for dataset in DATASETS:
            result = load_result(RUNS_DIR, model, dataset)

            for metric in QUAL_METRICS:
                m, s = result[metric]["mean"], result[metric]["std"]
                row = {"Model": model, "Dataset": dataset, "Metric": metric, "avg": m, "stdev": s}
                all_data.append(row)

                ref_hist = result[metric]["data_hist"]
                sample_hist = result[metric]["samples_hist"]

                if metric != 'nspdk':
                    ref_hist = to_histogram(ref_hist)
                    sample_hist = to_histogram(sample_hist)

                for i, (r, g) in enumerate(zip(ref_hist, sample_hist), 1):
                    row = {"Model": "Data", "Dataset": dataset, "Metric": metric, "Value": r}
                    all_hist_data.append(row)
                    row = {"Model": model, "Dataset": dataset, "Metric": metric, "Value": g}
                    all_hist_data.append(row)

            for metric in QUANT_METRICS:
                m = result[metric]
                row = {"Model": model, "Dataset": dataset, "Metric": metric, "avg": m, "stdev": None}
                all_data.append(row)

    all_data = pd.DataFrame(all_data)
    all_hist_data = pd.DataFrame(all_hist_data)
    return all_data, all_hist_data


def collate_result(result):
    # gens, refs = [], []
    # for elem in result:
    #     print(elem["model"], elem["dataset"], elem["metric"])
    elem = result[0]
    try:
        ref = elem["ref"].reshape(-1)
    except:
        ref = elem["ref"].toarray().reshape(-1)
    # refs.append(ref)

    try:
        gen = elem["gen"].reshape(-1)
    except:
        gen = elem["gen"].toarray().reshape(-1)
        # gens.append(gen)

    # print([l.shape for l in gens])
    # ref = np.vstack(refs).mean(axis=1)
    # gen = np.vstack(gens).mean(axis=1)

    return ref, gen



def collate_scores():
    SCORES_DIR = ROOT / "SCORES"
    rows = []

    for dataset in DATASET_NAMES:
        for model in MODEL_NAMES:
            for metric in QUAL_METRICS:
                path = SCORES_DIR / f"{model}_{dataset}_{metric}.pt"
                if path.exists():
                    ref, gen = collate_result(torch.load(path))
                    for i in range(len(ref)):
                        rows.append({
                            "Model": "Data",
                            "Dataset": dataset,
                            "Metric": metric,
                            "Value": ref[i]
                        })
                    for i in range(len(gen)):
                        rows.append({
                            "Model": model,
                            "Dataset": dataset,
                            "Metric": metric,
                            "Value": gen[i]
                        })

    all_scores = pd.DataFrame(rows)
    return all_scores[all_scores.Value>0]



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
                row = {"Order": order, "Dataset": dataset, "Metric": metric, "avg": m, "stdev": s}
                all_data.append(row)

    return pd.DataFrame(all_data)


HUMANIZE = {
    "PROTEINS_full": "PROTEINS",
    "trees": "TREES",
    "ladders": "LADDERS",
    "community": "COMMUNITY",
    "ego": "EGO",
    "ENZYMES": "ENZYMES",
    "bfs-fixed": "BFS",
    "dfs-fixed": "DFS",
    "random": "RANDOM",
    "dfs-random": "DFS RANDOM",
    "bfs-random": "BFS RANDOM",
    "smiles": "SMILES"
}



def parse_log(path):
    rows = []
    dataset, order = None, None
    with open(path, "r") as f:
        for l in f.readlines():
            if l.startswith("Training"):
                line = l.split(" ")
                dataset, order = HUMANIZE[line[2]], HUMANIZE[line[5].rsplit("\n")[0]]
            else:
                epoch, _, _, lt, _ = l.split(" - ")
                _, value = lt.split(": ")
                rows.append({
                    "Dataset": dataset,
                    "Order": order,
                    "Loss": float(value),
                    "Epoch": int(epoch)
                })

    return pd.DataFrame(rows)
