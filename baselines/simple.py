import numpy as np
import networkx as nx

import pyemd
from scipy.linalg import toeplitz

from dataset.graph import GraphList


def emd_distance(x, y, distance_scaling=1.0):
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(np.float)
    distance_mat = d_mat / distance_scaling

    # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float)
    y = y.astype(np.float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    emd = pyemd.emd(x, y, distance_mat)
    return emd


def loss(x, n, G, generator, metric):
    '''
    :param x: 1-D array, parameters to be optimized
    :param n: n for pred graph;
    :param G: real graph in networkx format;
    :param generator: 'BA', 'ER';
    :param metric: 'degree', 'clustering'
    :return: loss: emd distance
    '''
    # get real and pred graphs
    if generator == 'BA':
        S = nx.barabasi_albert_graph(n, int(np.rint(x)))
    elif generator == 'ER':
        S = nx.fast_gnp_random_graph(n, x)
    else:
        raise ValueError('Either generator="BA" or generator="er"')

    # define metric
    if metric == 'degree':
        G_hist = np.array(nx.degree_histogram(G))
        G_hist = G_hist / np.sum(G_hist)
        S_hist = np.array(nx.degree_histogram(S))
        S_hist = S_hist / np.sum(S_hist)

    elif metric == 'clustering':
        G_hist, _ = np.histogram(
            np.array(list(nx.clustering(G).values())), bins=50, range=(0.0, 1.0), density=False)
        G_hist = G_hist / np.sum(G_hist)
        S_hist, _ = np.histogram(
            np.array(list(nx.clustering(G).values())), bins=50, range=(0.0, 1.0), density=False)
        S_hist = S_hist / np.sum(S_hist)

    else:
        raise ValueError('Either metric="degree" or metric="clustering"')

    loss = emd_distance(G_hist, S_hist)
    return loss


def optimizer_brute(x_min, x_max, x_step, n, G, generator, metric):
    losses = []
    x_list = np.arange(x_min, x_max, x_step)

    for x_test in x_list:
        losses.append(loss(x_test, n, G, generator, metric))

    x_optim = x_list[np.argmin(np.array(losses))]
    return x_optim


def train_optimizationbased(graphlist, generator, metric):
    nodelist = graphlist.num_nodes()

    parameter = {}
    for G, nodes in zip(graphlist, nodelist):

        if generator == 'BA':
            n = nodes
            m = optimizer_brute(1, 10, 1, nodes, G, generator, metric)
            parameter_temp = [n, m, 1]
        elif generator == 'ER':
            n = nodes
            p = optimizer_brute(1e-6, 1, 0.01, nodes, G, generator, metric)
            parameter_temp = [n, p, 1]

        # update parameter list
        if nodes not in parameter.keys():
            parameter[nodes] = parameter_temp
        else:
            count = parameter[nodes][2]
            parameter[nodes] = [(parameter[nodes][i]*count+parameter_temp[i])/(count+1)
                                for i in range(len(parameter[nodes]))]
            parameter[nodes][2] = count+1

    return parameter


def sampler(nodelist, parameter, generator):
    graphs = []

    for nodes in nodelist:
        if nodes not in parameter.keys():
            nodes = min(parameter.keys(), key=lambda k: abs(k - nodes))
        if generator == 'BA':
            n = int(parameter[nodes][0])
            m = int(np.rint(parameter[nodes][1]))
            graph = nx.barabasi_albert_graph(n, m)
        if generator == 'ER':
            n = int(parameter[nodes][0])
            p = parameter[nodes][1]
            graph = nx.fast_gnp_random_graph(n, p)
        graphs.append(graph)

    return graphs


def run_baseline(generator, metric, graphlist):
    # show graphlist statistics
    print('total graph num: {}'.format(len(graphlist)))
    print('max number node: {}'.format(graphlist.max_nodes))

    parameter = train_optimizationbased(graphlist,
                                        generator=generator,
                                        metric=metric)

    samples = sampler(graphlist.num_nodes(),
                      parameter=parameter,
                      generator=generator)
    samples = GraphList(samples)
