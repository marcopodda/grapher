import torch
import numpy as np
import networkx as nx
from pathlib import Path
from joblib import Parallel, delayed
from functools import partial

from utils import mmd
from utils.serializer import load_yaml
from utils.constants import DATASET_NAMES, MODEL_NAMES, QUALITATIVE_METRIC_NAMES, RUNS_DIR, DATA_DIR
from utils.evaluation import orca


def pad(v, dim):
    pad_vec = np.zeros((dim,)) - 1
    pad_vec[:len(v)] = v
    return pad_vec


def patch(samples):
    for i, G in enumerate(samples):
        samples[i] = nx.convert_node_labels_to_integers(G)
        nodes = max(nx.connected_components(samples[i]), key=len)
        samples[i] = nx.Graph(samples[i]).subgraph(nodes)
    return samples


def degree_worker(G):
    return list(dict(nx.degree(G)).values())


def degree_dist(samples):
    P = Parallel(n_jobs=40, verbose=1)
    return P(delayed(degree_worker)(G) for G in samples)


def clustering_worker(G):
    return list(dict(nx.clustering(G)).values())


def clustering_dist(samples):
    P = Parallel(n_jobs=40, verbose=1)
    return P(delayed(clustering_worker)(G) for G in samples)


def orbit_worker(G):
    return orca(G)


def orbit_dist(samples):
    P = Parallel(n_jobs=40, verbose=1)
    return P(delayed(orbit_worker)(G) for G in samples)


def betweenness_worker(G):
    return list(dict(nx.betweenness_centrality(G)).values())


def betweenness_dist(samples):
    P = Parallel(n_jobs=40, verbose=1)
    return P(delayed(betweenness_worker)(G) for G in samples)


METRICS = {
    "degree": {"fun": degree_dist, "kwargs": dict(metric=mmd.gaussian_emd, is_hist=True, n_jobs=40)},
    "clustering": {"fun": clustering_dist, "kwargs": dict(metric=partial(mmd.gaussian_emd, sigma=0.1, distance_scaling=100), is_hist=True, n_jobs=40)},
    "orbit": {"fun": orbit_dist, "kwargs": dict(metric=partial(mmd.gaussian, sigma=30.0), is_hist=True, n_jobs=40)},
    "betweenness": {"fun": betweenness_dist, "kwargs": dict(metric=mmd.gaussian_emd, is_hist=True, n_jobs=40)},
    "nspdk": {"fun": betweenness_dist, "kwargs": dict(metric="nspdk", is_hist=False, n_jobs=40)},
}


def score(model, dataset, metric):
    raw_dir = DATA_DIR / dataset / "raw"
    data = torch.load(raw_dir / f"{dataset}.pt").graphlist
    splits = load_yaml(raw_dir / f"splits.yaml")
    test_set = [data[i] for i in splits["test"]]
    test_set = patch(test_set)
    max_test_nodes = max([G.number_of_nodes() for G in test_set])

    generated_dir = RUNS_DIR / model / dataset / "samples"

    scores = []
    for generated_path in generated_dir.iterdir():
        if '1000' in generated_path.stem or '5000' in generated_path.stem:
            continue
        generated = patch(torch.load(generated_path))

        fun = METRICS[metric]["fun"]
        mmd_kwargs = METRICS[metric]["kwargs"]

        gen_dist = fun(generated)
        test_dist = fun(test_set)
        score = mmd.compute_mmd(test_dist, gen_dist, **mmd_kwargs)
        print(model, dataset, metric, score)
        scores.append((score, gen_dist, test_dist))
    return scores


def score_all():
    SCORES_DIR = Path("SCORES")
    for model in MODEL_NAMES:
        for dataset in DATASET_NAMES:
            for metric in QUALITATIVE_METRIC_NAMES:
                if not (SCORES_DIR / f"{model}_{dataset}_{metric}.pt").exists():
                    s = score(model, dataset, metric)
                    torch.save(s, SCORES_DIR / f"{model}_{dataset}_{metric}.pt")