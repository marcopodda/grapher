import numpy as np
import networkx as nx
import subprocess as sp
from scipy.stats import entropy

from utils.misc import graph_to_file
from utils import mmd


BINS = 100
EPS =  + 1e-9


def degree_histogram(graphs):
    counts = []

    for G in graphs:
        degrees = dict(nx.degree(G))
        counts.extend(list(degrees.values()))

    return np.array(counts)


def clustering_histogram(graphs):
    coefs = []

    for G in graphs:
        clustering_coefs = dict(nx.clustering(G))
        coefs.extend(list(clustering_coefs.values()))

    return np.array(coefs)


def betweenness_histogram(graphs):
    counts = []

    for G in graphs:
        bcs = dict(nx.betweenness_centrality(G))
        counts.extend(list(bcs.values()))

    return np.array(counts)


def orbit_count_histogram(graphs):
    counts = []

    for G in graphs:
        graph_to_file(G, "./utils/orca/graph.in")
        sp.check_output(['./utils/orca/orca.exe', 'node', '4', './utils/orca/graph.in', './utils/orca/graph.out'])

        with open("./utils/orca/graph.out", "r") as f:
            count = []
            for line in f.readlines():
                line = line.rstrip("\n")
                line = [int(x) for x in line.split(" ")]
                count.append(sum(line))
        counts.extend(count)

    return np.array(counts)


def nspdk(ref, sample):
    sample = [G for G in sample if not G.number_of_nodes() == 0]
    mmd_dist = mmd.compute_mmd(ref, sample, metric='nspdk', is_hist=False, n_jobs=-1)
    return mmd_dist


def normalize_counts(ref_counts, sample_counts, bins):
    min_value = max(ref_counts.min(), sample_counts.min())
    max_value = max(ref_counts.max(), sample_counts.max())

    if max_value == 0:
        return np.zeros((bins,)), np.zeros((bins,))

    ref_counts = (ref_counts - min_value) / (max_value - min_value)
    ref_hist, _ = np.histogram(ref_counts, bins=bins, range=(0.0, 1.0), density=False)

    sample_counts = (sample_counts - min_value) / (max_value - min_value)
    sample_hist, _ = np.histogram(sample_counts, bins=bins, range=(0.0, 1.0), density=False)

    return ref_hist, sample_hist


def kl_divergence(ref, sample, metric):
    ref = [clean_graph(G_or_edges) for G_or_edges in ref]
    sample = [clean_graph(G_or_edges) for G_or_edges in sample]

    if metric == 'degree':
        ref_counts = degree_histogram(ref)
        sample_counts = degree_histogram(sample)

    elif metric == 'clustering':
        ref_counts = clustering_histogram(ref)
        sample_counts = clustering_histogram(sample)

    elif metric == 'orbit':
        ref_counts = orbit_count_histogram(ref)
        sample_counts = orbit_count_histogram(sample)

    elif metric == 'betweenness':
        ref_counts = betweenness_histogram(ref)
        sample_counts = betweenness_histogram(sample)

    ref_hist, sample_hist = normalize_counts(ref_counts, sample_counts, BINS)
    return entropy(ref_hist + EPS, sample_hist + EPS), ref_hist, sample_hist


def clean_graph(G_or_edges):
    if isinstance(G_or_edges, list) or isinstance(G_or_edges, tuple):
        G = nx.Graph(G_or_edges)
    else:
        G = G_or_edges
    G.remove_edges_from(nx.selfloop_edges(G))
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
    ref = [clean_graph(G_or_edges) for G_or_edges in ref]
    sample = [clean_graph(G_or_edges) for G_or_edges in sample]

    for i, G in enumerate(sample):
        if not is_duplicate(G, ref, fast):
            novel.append(G)

    if len(novel) == 0:
        return 0.0, []

    return len(novel) / len(sample), novel


def uniqueness(sample, fast):
    unique = []
    sample = [clean_graph(G_or_edges) for G_or_edges in sample]

    for i, G in enumerate(sample):
        if not is_duplicate(G, sample[i+1:], fast):
            unique.append(G)

    if len(unique) == 0:
        return 0.0, []

    return len(unique) / len(sample), unique
