import os
from eden.graph import vectorize
import numpy as np
import itertools
import networkx as nx
import subprocess as sp
from scipy.stats import entropy
from joblib import Parallel, delayed
import tempfile

from utils.misc import graph_to_file


BINS = 100
EPS =  + 1e-9


def degree_worker(G):
    degrees = dict(nx.degree(G))
    return list(degrees.values())


def degree_histogram(graphs):
    P = Parallel(n_jobs=-1)
    counts = P(delayed(degree_worker)(G) for G in graphs)
    counts = list(itertools.chain.from_iterable(counts))
    return np.array(counts)


def clustering_worker(G):
    clustering_coefs = dict(nx.clustering(G))
    clustering_coefs = list(clustering_coefs.values())
    hist, _ = np.histogram(clustering_coefs, bins=100, range=(0.0, 1.0), density=False)
    return hist


def clustering_histogram(graphs):
    P = Parallel(n_jobs=-1)
    counts = P(delayed(clustering_worker)(G) for G in graphs)
    counts = list(itertools.chain.from_iterable(counts))
    return np.array(counts)


def betweenness_worker(G):
    bcs = dict(nx.betweenness_centrality(G))
    return list(bcs.values())


def betweenness_histogram(graphs):
    P = Parallel(n_jobs=-1)
    counts = P(delayed(betweenness_worker)(G) for G in graphs)
    counts = list(itertools.chain.from_iterable(counts))
    return np.array(counts)


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    COUNT_START_STR = 'orbit counts: \n'
    tmp_fname = tempfile.NamedTemporaryFile().name

    f = open(tmp_fname, 'w')
    f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()

    output = sp.check_output(['./utils/orca/orca', 'node', '4', tmp_fname, 'std'])
    output = output.decode('utf8').strip()

    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR)
    output = output[idx:]
    node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ')))
                                  for node_cnts in output.strip('\n').split('\n')])

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts


def orbit_worker(graph):
    try:
        orbit_counts = orca(graph)
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / graph.number_of_nodes()
        return orbit_counts_graph
    except Exception as e:
        return np.zeros((graph.number_of_nodes(), 15))


def orbit_count_histogram(graphs):
    P = Parallel(n_jobs=-1)
    counts = P(delayed(orbit_worker)(G) for G in graphs)
    return np.array(counts)


def nspdk(graphs):
    for i, G in enumerate(graphs):
        graphs[i] = nx.convert_node_labels_to_integers(G)

    return vectorize(graphs)


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

    elif metric == 'nspdk':
        ref_counts = nspdk(ref)
        sample_counts = nspdk(sample)

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
            test = nx.faster_could_be_isomorphic(G, g)

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
