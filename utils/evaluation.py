import numpy as np
import networkx as nx
from scipy.stats import entropy

from .graphlets import graphlet_count

BINS = 30
EPS = 1e-8


def _get_hist(graphs, func, bins=BINS):
    rng = (0.0, 1.0)
    hists = []

    for G in graphs:
        values = np.array(list(dict(func(G)).values()))
        if values.max() > rng[1]:
            values = values / values.sum()
        hist, _ = np.histogram(values, bins=bins, range=range, density=False)
        hists.append(hist)
    return hists


def get_dd_hists(graphs, bins=BINS, range=(0.0, 1.0)):
    return _get_hist(graphs, nx.degree, bins)


def get_cc_hists(graphs, bins=BINS, range=(0.0, 1.0)):
    return _get_hist(graphs, nx.clustering, bins)


def get_gl_hists(graphs, bins=BINS, range=(0.0, 1.0)):
    hists = []

    for G in graphs:
        counts = graphlet_count(G)
        values = list(counts.values())
        hist, _ = np.histogram(values, bins=bins, range=range)
        hists.append(hist)

    return hists


def pad_vectors(sample_ref, sample_pred, bin_size):
    counts_ref = np.zeros((bin_size, )) + EPS
    counts_pred = np.zeros((bin_size, )) + EPS

    for hist in sample_ref:
        counts_ref[:hist.shape[0]] += np.array(hist)

    for hist in sample_pred:
        counts_pred[:hist.shape[0]] += np.array(hist)

    return counts_ref, counts_pred


def kl_divergence(sample_ref, sample_pred, bin_size):
    counts_ref, counts_pred = pad_vectors(sample_ref, sample_pred, bin_size)
    return entropy(counts_ref, counts_pred), counts_ref, counts_pred


def graphlet_kl(ref, target):
    if isinstance(ref[0], tuple) or isinstance(ref[0], list):
        ref = [clean_graph(e) for e in ref]

    if isinstance(target[0], tuple) or isinstance(target[0], list):
        target = [clean_graph(e) for e in target]

    bins = 30
    gl_hist_ref = get_gl_hists(ref, bins=bins)
    gl_hist_pred = get_gl_hists(target, bins=bins)
    return kl_divergence(gl_hist_ref, gl_hist_pred, bins)


def clustering_kl(ref, target):
    if isinstance(ref[0], tuple) or isinstance(ref[0], list):
        ref = [clean_graph(e) for e in ref]

    if isinstance(target[0], tuple) or isinstance(target[0], list):
        target = [clean_graph(e) for e in target]

    clust_hist_ref = get_cc_hists(ref, bins=BINS)
    clust_hist_pred = get_cc_hists(target, bins=BINS)
    return kl_divergence(clust_hist_ref, clust_hist_pred, BINS)


def degree_kl(ref, target):
    if isinstance(ref[0], tuple) or isinstance(ref[0], list):
        ref = [clean_graph(e) for e in ref]

    if isinstance(target[0], tuple) or isinstance(target[0], list):
        target = [clean_graph(e) for e in target]

    deg_hist_ref = get_dd_hists(ref)
    deg_hist_pred = get_dd_hists(target)
    max_num_ref = max([G.number_of_nodes() for G in ref])
    max_num_pred = max([G.number_of_nodes() for G in target])
    max_num = max(max_num_ref, max_num_pred)
    return kl_divergence(deg_hist_ref, deg_hist_pred, max_num)



def clean_graph(G_or_edges):
    if isinstance(G_or_edges, list) or isinstance(G_or_edges, tuple):
        G = nx.Graph(G_or_edges)
    return max(nx.connected_component_subgraphs(G), key=len)


def find_duplicates(G, Gs):
    for g in Gs:
        if nx.is_isomorphic(G, g):
            return True

    return False


def novelty(ref, target):
    if isinstance(ref[0], tuple) or isinstance(ref[0], list):
        ref = [clean_graph(e) for e in ref]

    if isinstance(target[0], tuple) or isinstance(target[0], list):
        target = [clean_graph(e) for e in target]

    res = []

    for G in ref:
        res.append(find_duplicates(G, target))

    return 1 - sum(res) / len(ref)


def uniqueness(target):
    if isinstance(target[0], tuple) or isinstance(target[0], list):
        target = [clean_graph(e) for e in target]

    res = []

    for i, G in enumerate(target):
        new_target = target[:i] + target[i + 1:]
        res.append(find_duplicates(G, new_target))

    return 1 - sum(res) / len(target)