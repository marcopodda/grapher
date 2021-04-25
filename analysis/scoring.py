import time
import torch
import numpy as np
import networkx as nx
from pathlib import Path
from joblib import Parallel, delayed
from functools import partial

from utils import mmd
from utils.serializer import load_yaml
from utils.constants import DATASET_NAMES, MODEL_NAMES, QUALITATIVE_METRIC_NAMES, RUNS_DIR, DATA_DIR
from utils.evaluation import orca, nspdk


def patch(samples):
    for i, G in enumerate(samples):
        samples[i] = nx.convert_node_labels_to_integers(G)
        nodes = max(nx.connected_components(samples[i]), key=len)
        samples[i] = nx.Graph(samples[i]).subgraph(nodes)
    return samples


def degree_worker(G):
    degree_dist = list(dict(nx.degree(G)).values())
    return np.array(degree_dist)


def degree_dist(samples):
    P = Parallel(n_jobs=40, verbose=0)
    counts = P(delayed(degree_worker)(G) for G in samples)
    return np.array(counts)


def clustering_worker(G):
    clustering_coefs = list(dict(nx.clustering(G)).values())
    hist, _ = np.histogram(clustering_coefs, bins=100, range=(0.0, 1.0), density=False)
    return hist


def clustering_dist(samples):
    P = Parallel(n_jobs=40, verbose=0)
    counts = P(delayed(clustering_worker)(G) for G in samples)
    return np.array(counts)


def orbit_worker(graph):
    try:
        orbit_counts = orca(graph)
        counts = np.sum(orbit_counts, axis=0) / graph.number_of_nodes()
        return counts
    except Exception as e:
        return np.zeros((graph.number_of_nodes(), 15))


def orbit_dist(graphs):
    P = Parallel(n_jobs=40, verbose=0)
    counts = P(delayed(orbit_worker)(G) for G in graphs)
    return np.array(counts)


def betweenness_worker(G):
    bcs = list(dict(nx.betweenness_centrality(G)).values())
    return np.array(bcs)


def betweenness_dist(samples):
    P = Parallel(n_jobs=40, verbose=0)
    counts = P(delayed(betweenness_worker)(G) for G in samples)
    return np.array(counts)


METRICS = {
    "degree": {"fun": degree_dist, "kwargs": dict(metric=mmd.gaussian_emd, is_hist=True, n_jobs=40)},
    "clustering": {"fun": clustering_dist, "kwargs": dict(metric=partial(mmd.gaussian_emd, sigma=0.1, distance_scaling=100), is_hist=True, n_jobs=40)},
    "orbit": {"fun": orbit_dist, "kwargs": dict(metric=partial(mmd.gaussian_emd, sigma=30.0), is_hist=True, n_jobs=40)},
    "betweenness": {"fun": betweenness_dist, "kwargs": dict(metric=mmd.gaussian_emd, is_hist=True, n_jobs=40)},
    "nspdk": {"fun": nspdk, "kwargs": dict(metric="nspdk", is_hist=False, n_jobs=40)},
}


def load_test_set(dataset):
    raw_dir = DATA_DIR / dataset / "raw"
    data = torch.load(raw_dir / f"{dataset}.pt").graphlist
    splits = load_yaml(raw_dir / f"splits.yaml")
    test_set = [data[i] for i in splits["test"]]
    return patch(test_set)


def load_samples(path):
    samples = torch.load(path)
    return patch([G for G in samples if not G.number_of_nodes() == 0])


def score(test_set, model, dataset, metric):
    generated_dir = RUNS_DIR / model / dataset / "samples"

    scores = []

    for i, generated_path in enumerate(generated_dir.iterdir()):
        if '1000' in generated_path.stem or '5000' in generated_path.stem:
            continue

        if i == 3:
            break

        start = time.time()

        generated = load_samples(generated_path)
        fun = METRICS[metric]["fun"]
        mmd_kwargs = METRICS[metric]["kwargs"]

        gen_dist = fun(generated)
        test_dist = fun(test_set)
        score = mmd.compute_mmd(test_dist, gen_dist, **mmd_kwargs)

        elapsed = time.time() - start
        print(f"{model} {dataset} {metric} {i} {elapsed}")

        scores.append({
            "model": model,
            "dataset": dataset,
            "metric": metric,
            "score": score,
            "gen": gen_dist,
            "ref": test_dist})
    return scores


def score_all():
    SCORES_DIR = Path("SCORES")
    for dataset in DATASET_NAMES:
        test_set = load_test_set(dataset)
        for model in MODEL_NAMES:
            for metric in QUALITATIVE_METRIC_NAMES:
                if not (SCORES_DIR / f"{model}_{dataset}_{metric}.pt").exists():
                    s = score(test_set, model, dataset, metric)
                    torch.save(s, SCORES_DIR / f"{model}_{dataset}_{metric}.pt")