import itertools
import subprocess as sp
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
from utils.graphs import max_connected_comp
from utils.misc import graph_to_file
from joblib import Parallel, delayed


def reorder(obj):
    indices = sorted([o[0] for o in obj])
    return list(itertools.chain.from_iterable([obj[i][1] for i in indices]))


def patch(samples):
    return [nx.convert_node_labels_to_integers(G) for G in samples]


def pad(v, maxlen=100):
    w = torch.zeros(maxlen)
    w[:len(v)] = v
    return w.tolist()


def degree_worker(G):
    return pad(nx.degree_histogram(G))


def degree_dist(samples, n_jobs=40):
    P = Parallel(n_jobs=n_jobs, verbose=0)
    counts = P(delayed(degree_worker)(G) for (i, G) in enumerate(samples))
    counts = list(itertools.chain.from_iterable(counts))
    return np.array(counts)


def clustering_worker(G):
    clustering_coefs = dict(nx.clustering(G))
    clustering_coefs = list(clustering_coefs.values())
    hist, _ = np.histogram(clustering_coefs, bins=100, range=(0.0, 1.0), density=False)
    return hist


def clustering_dist(samples, n_jobs=40):
    P = Parallel(n_jobs=n_jobs, verbose=0)
    counts = P(delayed(clustering_worker)(G) for (i, G) in enumerate(samples))
    counts = list(itertools.chain.from_iterable(counts))
    return np.array(counts)


def orbit_worker(i, G):
    try:
        counts = orca(G)
        counts = counts.sum(axis=1)
        return i, counts
    except Exception as e:
        return i, [0]


def orbit_dist(samples, n_jobs=40):
    P = Parallel(n_jobs=n_jobs, verbose=0)
    counts = P(delayed(orbit_worker)(i, G) for (i, G) in enumerate(samples))
    counts = list(itertools.chain.from_iterable(counts))
    return np.array(counts)


def betweenness_worker(G):
    betweenness = dict(nx.betweenness_centrality(G))
    betweenness = list(betweenness.values())
    hist, _ = np.histogram(betweenness, bins=100, range=(0.0, 1.0), density=False)
    return hist


def betweenness_dist(samples, n_jobs=40):
    P = Parallel(n_jobs=n_jobs, verbose=0)
    counts = P(delayed(betweenness_worker)(i, G) for (i, G) in enumerate(samples))
    counts = list(itertools.chain.from_iterable(counts))
    return np.array(counts)


def random_sample(graphs, n=100):
    if n >= len(graphs):
        return graphs
    indices = np.random.choice(len(graphs), n, replace=False)
    return [graphs[i] for i in indices]


def clean_graph(G):
    G = max_connected_comp(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.relabel_nodes(G, {n: i for (i, n) in enumerate(G.nodes())})
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
    return 1.0 - sum(counts) / len(counts)


def uniqueness(samples, fast):
    samples = [clean_graph(G_or_edges) for G_or_edges in samples]

    P = Parallel(n_jobs=48, verbose=0)
    counts = P(delayed(dup)(G, samples[i+1:], fast) for i, G in enumerate(samples[:-1]))
    return 1.0 - sum(counts) / len(counts)


def normalize(ref_counts, gen_counts, bins=100, norm=True):
    if norm:
        m1, m2 = min(ref_counts.min(), gen_counts.min()), max(ref_counts.max(), gen_counts.max())
        ref_counts = (ref_counts - m1) / (m2 - m1 + 1e-8)
        gen_counts = (gen_counts - m1) / (m2 - m1 + 1e-8)
    ref_hist, _ = np.histogram(ref_counts, bins=bins, range=(0.0, 1.0), density=False)
    gen_hist, _ = np.histogram(gen_counts, bins=bins, range=(0.0, 1.0), density=False)
    return ref_hist, gen_hist