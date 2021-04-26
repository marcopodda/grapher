import time
import torch
import numpy as np
import networkx as nx
from pathlib import Path
from joblib import Parallel, delayed
from functools import partial
from eden.graph import vectorize

from utils import mmd
from utils.serializer import load_yaml
from utils.constants import RUNS_DIR, DATA_DIR
from utils.evaluation import orca
from joblib import Parallel, delayed


def patch(samples):
    graphs = []
    for G in samples:
        G = nx.convert_node_labels_to_integers(G)
        nodes = max(nx.connected_components(G), key=len)
        G = nx.Graph(G).subgraph(nodes)
        graphs.append(G)
    return graphs


def pad(lvec):
    width = max(len(l) for l in lvec)
    height = len(lvec)
    mat = np.zeros((height, width))
    for i, v in enumerate(lvec):
        mat[i, :len(v)] = v
    return mat


def degree_worker(G):
    return list(dict(nx.degree(G)).values())


def degree_dist(samples):
    P = Parallel(n_jobs=40, verbose=0)
    counts = P(delayed(degree_worker)(G) for G in samples)
    return pad(counts)


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


def orbit_dist(samples):
    P = Parallel(n_jobs=40, verbose=0)
    counts = P(delayed(orbit_worker)(G) for G in samples)
    return np.array(counts)


def betweenness_worker(G):
    return list(dict(nx.betweenness_centrality(G)).values())


def betweenness_dist(samples):
    P = Parallel(n_jobs=40, verbose=0)
    counts = P(delayed(betweenness_worker)(G) for G in samples)
    return pad(counts)


def nspdk_dist(samples):
    for i, G in enumerate(samples):
        samples[i] = nx.convert_node_labels_to_integers(G)

    counts = vectorize(samples, complexity=4, discrete=True)
    return counts


def random_sample(graphs, n=100):
    if n >= len(graphs):
        return graphs
    indices = np.random.choice(len(graphs), n)
    return [graphs[i] for i in indices]


def load_test_set(dataset):
    raw_dir = DATA_DIR / dataset / "raw"
    data = torch.load(raw_dir / f"{dataset}.pt").graphlist
    splits = load_yaml(raw_dir / f"splits.yaml")
    test_set = [data[i] for i in splits["test"]]
    return patch(test_set)


def load_samples(path):
    samples = torch.load(path)
    samples = [G for G in samples if not G.number_of_nodes() == 0]
    return patch(samples)


def clean_graph(G_or_edges):
    if isinstance(G_or_edges, list) or isinstance(G_or_edges, tuple):
        G = nx.Graph(G_or_edges)
    else:
        G = G_or_edges

    nodes = max(nx.connected_components(G), key=len)
    G = nx.Graph(G.subgraph(nodes))
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def novelty_worker(G, ref):
    count = np.array([nx.is_isomorphic(G, g) for g in ref])
    return count.sum() > 0


def novelty(ref, samples):
    ref = [clean_graph(G_or_edges) for G_or_edges in ref]
    samples = [clean_graph(G_or_edges) for G_or_edges in samples]

    P = Parallel(n_jobs=3, verbose=0)
    counts = P(delayed(novelty_worker)(G, ref) for G in samples)
    return sum(counts) / len(counts)


def uniqueness_worker(G, samples):
    count = np.array([nx.is_isomorphic(G, g) for g in samples])
    return count.sum() > 0


def uniqueness(samples):
    samples = [clean_graph(G_or_edges) for G_or_edges in samples]

    P = Parallel(n_jobs=3, verbose=0)
    counts = P(delayed(uniqueness_worker)(G, samples[i+1:]) for i, G in enumerate(samples[:-1]))
    return sum(counts) / len(counts)