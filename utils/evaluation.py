import numpy as np
import networkx as nx
from scipy.stats import entropy

BINS = 100
EPS = 1e-8


def get_dd_hists(graphs):
    degree_hists = []
    for G in graphs:
        hist = np.array(nx.degree_histogram(G))
        degree_hists.append(hist)
    return degree_hists


def get_cc_hists(graphs, bins=BINS, range=(0.0, 1.0)):
    cc_hists = []
    for G in graphs:
        coefs = list(nx.clustering(G).values())
        hist, _ = np.histogram(coefs, bins=bins, range=range, density=False)
        cc_hists.append(hist)
    return cc_hists


def clustering_kl(graph_ref, graph_pred):
    clust_hist_ref = get_cc_hists(graph_ref, bins=BINS)
    clust_hist_pred = get_cc_hists(graph_pred, bins=BINS)
    return kl_divergence(clust_hist_ref, clust_hist_pred, BINS)


def degree_kl(graph_ref, graph_pred):
    deg_hist_ref = get_dd_hists(graph_ref)
    deg_hist_pred = get_dd_hists(graph_pred)
    max_num_ref = max([G.number_of_nodes() for G in graph_ref])
    max_num_pred = max([G.number_of_nodes() for G in graph_pred])
    max_num = max(max_num_ref, max_num_pred)
    return kl_divergence(deg_hist_ref, deg_hist_pred, max_num)


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
    return entropy(counts_ref, counts_pred)


def compute_statistics(graph_ref, graph_pred):
    kl_degree = degree_kl(graph_ref, graph_pred)
    kl_clust = clustering_kl(graph_ref, graph_pred)
    return kl_degree, kl_clust
