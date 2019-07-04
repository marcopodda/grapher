import numpy as np
import networkx as nx
from scipy.stats import entropy

from .graphlets import graphlet_count

BINS = 30


def _get_hist(graphs, func):
    rng = (0.0, 1.0)
    hists = []

    for G in graphs:
        values = np.array(list(dict(func(G)).values()))
        if values.max() > rng[1]:
            values = values / values.sum()
        hist, _ = np.histogram(values, bins=BINS, range=rng, density=False)
        hists.append(hist)
    return np.array(hists).sum(axis=0)


def kl_divergence(ref, sample, metric):
    if isinstance(ref[0], tuple) or isinstance(ref[0], list):
        ref = [clean_graph(e) for e in ref]

    if isinstance(sample[0], tuple) or isinstance(sample[0], list):
        sample = [clean_graph(e) for e in sample]

    metric_fun = {
        'clustering': nx.clustering,
        'degree': nx.degree,
        'graphlet': graphlet_count
    }[metric]

    ref_hist = _get_hist(ref, metric_fun)
    sample_hist = _get_hist(sample, metric_fun)
    return entropy(ref_hist, sample_hist), ref_hist, sample_hist


def clean_graph(G_or_edges):
    if isinstance(G_or_edges, list) or isinstance(G_or_edges, tuple):
        G = nx.Graph(G_or_edges)
    return max(nx.connected_component_subgraphs(G), key=len)


def find_duplicates(G, Gs):
    for g in Gs:
        if nx.is_isomorphic(G, g):
            return True

    return False


def novelty(ref, sample):
    if isinstance(ref[0], tuple) or isinstance(ref[0], list):
        ref = [clean_graph(e) for e in ref]

    if isinstance(sample[0], tuple) or isinstance(sample[0], list):
        sample = [clean_graph(e) for e in sample]

    res = []

    for G in ref:
        res.append(find_duplicates(G, sample))

    return 1 - sum(res) / len(ref)


def uniqueness(sample):
    if isinstance(sample[0], tuple) or isinstance(sample[0], list):
        sample = [clean_graph(e) for e in sample]

    res = []

    for i, G in enumerate(sample):
        new_sample = sample[:i] + sample[i + 1:]
        res.append(find_duplicates(G, new_sample))

    return 1 - sum(res) / len(sample)