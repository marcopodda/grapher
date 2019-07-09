import numpy as np
import networkx as nx
from scipy.stats import entropy

from .graphlets import graphlet_count, average_graphlet_count

BINS = 100


def _get_hist(graphs, func, rng):
    hists = np.zeros((BINS,))

    for G in graphs:
        values = np.array(list(dict(func(G)).values()))
        hist, _ = np.histogram(values, bins=BINS, range=rng, density=False)
        hists += hist

    return hists / hists.sum()


# def average_graphlet_count(graphlist):
#     print(len(graphlist))
#     counts = []
#     for i, G in graphlist:
#         print(i)
#         values = np.array(list(dict(graphlet_count(G)).values()))
#         counts.append(values.sum(axis=0))
#     counts = np.array(counts)
#     return counts / counts.sum()


def kl_divergence(ref, sample, metric):
    print(metric)
    if isinstance(ref[0], tuple) or isinstance(ref[0], list):
        ref = [clean_graph(e) for e in ref]

    if isinstance(sample[0], tuple) or isinstance(sample[0], list):
        sample = [clean_graph(e) for e in sample]

    eps =  + 1e-9
    metric_fun, rng = {
        'clustering': (nx.clustering, (0.0, 1.0)),
        'degree': (nx.degree, (0.0, 100.0)),
        'graphlet': (graphlet_count, (0.0, 1000.0)),
    }[metric]

    if metric == "graphlet":
        ref_hist = average_graphlet_count(ref)
        sample_hist = average_graphlet_count(sample)
    else:
        ref_hist = _get_hist(ref, metric_fun, rng)
        sample_hist = _get_hist(sample, metric_fun, rng)
    print(len(ref_hist), len(sample_hist))
    return entropy(ref_hist + eps, sample_hist + eps), ref_hist, sample_hist


def clean_graph(G_or_edges):
    if isinstance(G_or_edges, list) or isinstance(G_or_edges, tuple):
        G = nx.Graph(G_or_edges)
    return G


def is_duplicate(G, Gs, fast):
    for g in Gs:
        if fast:
            mapping = {n: i for (i, n) in enumerate(g.nodes(), 3)}
            g = nx.relabel_nodes(g, mapping)
            test = sorted(G.edges()) == sorted(g.edges())
        else:
            test = nx.is_isomorphic(G, g)

        if test is True:
            return True

    return False


def novelty(ref, sample, fast):
    novel = []
    for i, G in enumerate(sample):
        if not is_duplicate(G, ref, fast):
            novel.append(G)

    if len(novel) == 0:
        return 0.0, []

    return len(novel) / len(sample), novel


def uniqueness(sample, fast):
    unique = []
    for i, G in enumerate(sample):
        if not is_duplicate(G, sample[i+1:], fast):
            unique.append(G)

    if len(unique) == 0:
        return 0.0, []

    return len(unique) / len(sample), unique


def filter_unique_and_novel(ref, sample, fast):
    _, novel = novelty(ref, sample, fast)
    _, unique = uniqueness(novel, fast)
    return unique
