import itertools
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


def degree_worker(G):
    degrees = dict(nx.degree(G))
    degrees = list(degrees.values())
    return degrees


def degree_dist(samples, n_jobs=40):
    P = Parallel(n_jobs=n_jobs, verbose=0)
    counts = P(delayed(degree_worker)(G) for G in samples)
    counts = list(itertools.chain.from_iterable(counts))
    return np.array(counts)


def clustering_worker(G):
    clustering_coefs = dict(nx.clustering(G))
    return list(clustering_coefs.values())


def clustering_dist(samples, n_jobs=40):
    P = Parallel(n_jobs=n_jobs, verbose=0)
    counts = P(delayed(clustering_worker)(G) for G in samples)
    counts = list(itertools.chain.from_iterable(counts))
    return np.array(counts)


def orbit_worker(G):
    try:
        counts = orca(G)
        counts = counts.sum(axis=1).reshape(-1)
        print(counts)
        return counts.tolist()
    except Exception as e:
        print("orca", e)
        return np.zeros(G.number_of_nodes)


def orbit_dist(samples, n_jobs=40):
    P = Parallel(n_jobs=n_jobs, verbose=0)
    counts = P(delayed(orbit_worker)(G) for G in samples)
    counts = list(itertools.chain.from_iterable(counts))
    return np.array(counts)


def betweenness_worker(G):
    betweenness = dict(nx.betweenness_centrality(G))
    return list(betweenness.values())


def betweenness_dist(samples, n_jobs=40):
    P = Parallel(n_jobs=n_jobs, verbose=0)
    counts = P(delayed(betweenness_worker)(G) for G in samples)
    counts = list(itertools.chain.from_iterable(counts))
    return np.array(counts)


def nspdk_dist(samples):
    for i, G in enumerate(samples):
        samples[i] = nx.convert_node_labels_to_integers(G)

    counts = vectorize(samples, complexity=4, discrete=True).toarray()
    print("nspdk", counts.shape)
    return counts.reshape(-1)


def random_sample(graphs, n=100):
    if n >= len(graphs):
        return graphs
    indices = np.random.choice(len(graphs), n, replace=False)
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
    G = nx.relabel_nodes(G, {n:i for (i,n) in enumerate(G.nodes())})
    return G


def dup(G, Gs, fast):
    test = False

    for g in Gs:
        test = sorted(G.edges()) == sorted(g.edges()) if fast else nx.faster_could_be_isomorphic(G, g)
        if test is True:
            break

    return test


def novelty(ref, samples, fast):
    ref = [clean_graph(G_or_edges) for G_or_edges in ref]
    samples = [clean_graph(G_or_edges) for G_or_edges in samples]

    P = Parallel(n_jobs=48, verbose=0)
    counts = P(delayed(dup)(G, ref, fast) for G in samples)
    return sum(counts) / len(counts)


def uniqueness(samples, fast):
    samples = [clean_graph(G_or_edges) for G_or_edges in samples]

    P = Parallel(n_jobs=48, verbose=0)
    counts = P(delayed(dup)(G, samples[i+1:], fast) for i, G in enumerate(samples[:-1]))
    return sum(counts) / len(counts)


def normalize(ref_counts, gen_counts, bins=100):
    ref_counts, _ = np.histogram(ref_counts, bins=bins, range=(0.0, 1.0), density=False)
    gen_counts, _ = np.histogram(gen_counts, bins=bins, range=(0.0, 1.0), density=False)
    return ref_counts, gen_counts