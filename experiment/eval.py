import itertools
import numpy as np
import networkx as nx
from joblib import Parallel, delayed

from utils.evaluation import orca
from utils.graphs import max_connected_comp
from joblib import Parallel, delayed


def patch(samples):
    return [nx.convert_node_labels_to_integers(G) for G in samples]


def pad(v, maxlen=100):
    w = np.zeros(maxlen)
    w[:len(v)] = v
    return w.tolist()


def degree_worker(G):
    return pad(nx.degree_histogram(G))


def degree_dist(samples, n_jobs=40):
    P = Parallel(n_jobs=n_jobs, verbose=0)
    counts = P(delayed(degree_worker)(G) for G in samples)
    return np.sum(counts, axis=0)


def clustering_worker(G):
    clustering_coefs = dict(nx.clustering(G))
    clustering_coefs = list(clustering_coefs.values())
    hist, _ = np.histogram(clustering_coefs, bins=100, range=(0.0, 1.0), density=False)
    return hist


def clustering_dist(samples, n_jobs=40):
    P = Parallel(n_jobs=n_jobs, verbose=0)
    counts = P(delayed(clustering_worker)(G) for G in samples)
    return np.sum(counts, axis=0)


def orbit_worker(G):
    try:
        counts = orca(G)
        counts = counts.sum(axis=0)
        return counts.tolist()
    except Exception as e:
        return [0]


def orbit_dist(samples, n_jobs=40):
    P = Parallel(n_jobs=n_jobs, verbose=0)
    counts = P(delayed(orbit_worker)(G) for G in samples)
    return np.sum(counts, axis=0)


def betweenness_worker(G):
    betweenness = dict(nx.betweenness_centrality(G))
    betweenness = list(betweenness.values())
    hist, _ = np.histogram(betweenness, bins=100, range=(0.0, 1.0), density=False)
    return hist


def betweenness_dist(samples, n_jobs=40):
    P = Parallel(n_jobs=n_jobs, verbose=0)
    counts = P(delayed(betweenness_worker)(G) for G in samples)
    return np.sum(counts, axis=0)


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