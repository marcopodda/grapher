import numpy as np
import networkx as nx
from scipy.stats import entropy

from .graphlets import graphlet_count

BINS = 100


def _get_hist(graphs, func, rng):
    hists = np.zeros((BINS,))

    for G in graphs:
        values = np.array(list(dict(func(G)).values()))
        hist, _ = np.histogram(values, bins=BINS, range=rng, density=False)
        hists += hist

    return hists / hists.sum()


def kl_divergence(ref, sample, metric):
    if isinstance(ref[0], tuple) or isinstance(ref[0], list):
        ref = [clean_graph(e) for e in ref]

    if isinstance(sample[0], tuple) or isinstance(sample[0], list):
        sample = [clean_graph(e) for e in sample]

    eps =  + 1e-9
    metric_fun, rng = {
        'clustering': (nx.clustering, (0.0, 1.0)),
        'degree': (nx.degree, (0.0, 100.0)),
        'graphlet': (graphlet_count, (0.0, 100.0))
    }[metric]

    ref_hist = _get_hist(ref, metric_fun, rng)
    sample_hist = _get_hist(sample, metric_fun, rng)
    return entropy(ref_hist + eps, sample_hist + eps), ref_hist, sample_hist


def clean_graph(G_or_edges):
    if isinstance(G_or_edges, list) or isinstance(G_or_edges, tuple):
        G = nx.Graph(G_or_edges)
    return G # max(nx.connected_component_subgraphs(G), key=len)


def is_duplicate(G, Gs):
    edges = sorted(G.edges())

    for g in Gs:
        if edges == sorted(g.edges()):
        # if nx.is_isomorphic(G, g):
            return True

    return False


def novelty(ref, sample):
    res = []
    for G in ref:
        if not is_duplicate(G, sample):
            res.append(G)

    return len(res) / len(ref), res


def is_empty(G):
    return list(G.nodes()) == []


def empty_graph():
    return nx.Graph()


def uniqueness(sample):
    res = []
    for i, G in enumerate(res):
        non_empty = [G for G in res if not is_empty(G)]
        if is_duplicate(G, non_empty):
            res[i] = empty_graph()

    unique = [G for G in res if not is_empty(G)]
    return len(unique) / len(sample), unique


def filter_unique_and_novel(ref, sample):
    _, novel = novelty(ref, sample)
    _, unique = uniqueness(novel)
    return unique
