import numpy as np
import networkx as nx

import pyemd
from scipy.linalg import toeplitz

from dataset.graph import GraphList
from joblib import Parallel, delayed


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


def loss(x, n, G_real, generator):
    '''
    :param x: 1-D array, parameters to be optimized
    :param
    n: n for pred graph;
    G: real graph in networkx format;
    generator: 'BA', 'Gnp', 'Powerlaw';
    metric: 'degree', 'clustering'
    :return: Loss: emd distance
    '''
    # get argument

    # get real and pred graphs
    if generator == 'BA':
        G_pred = nx.barabasi_albert_graph(n, int(np.rint(x)))
    if generator == 'ER':
        G_pred = nx.fast_gnp_random_graph(n, x)

    G_real_hist = np.array(nx.degree_histogram(G_real))
    G_real_hist = G_real_hist / np.sum(G_real_hist)
    G_pred_hist = np.array(nx.degree_histogram(G_pred))
    G_pred_hist = G_pred_hist/np.sum(G_pred_hist)
    loss1 = emd_distance(G_real_hist, G_pred_hist)

    G_real_hist, _ = np.histogram(
        np.array(list(nx.clustering(G_real).values())), bins=50, range=(0.0, 1.0), density=False)
    G_real_hist = G_real_hist / np.sum(G_real_hist)
    G_pred_hist, _ = np.histogram(
        np.array(list(nx.clustering(G_pred).values())), bins=50, range=(0.0, 1.0), density=False)
    G_pred_hist = G_pred_hist / np.sum(G_pred_hist)

    loss2 = emd_distance(G_real_hist, G_pred_hist)
    return loss1 + loss2


def optimizer_brute(x_min, x_max, x_step, n, G_real, generator):
    loss_all = []
    x_list = np.arange(x_min, x_max, x_step)
    P = Parallel(n_jobs=40, verbose=0)
    loss_all = P(delayed(loss)(x_test, n, G_real, generator) for x_test in x_list if x_test < n)
    # for x_test in x_list:
    #     if x_test < n:
    #         loss_all.append(loss(x_test, n, G_real, generator))
    x_optim = x_list[np.argmin(np.array(loss_all))]
    return x_optim


def train_optimizationbased(graphlist, generator):
    nodelist = graphlist.num_nodes()

    parameter = {}
    for G, nodes in zip(graphlist, nodelist):

        if generator == 'BA':
            n = nodes
            m = optimizer_brute(1, 10, 1, nodes, G, generator)
            parameter_temp = [n, m, 1]
        elif generator == 'ER':
            n = nodes
            p = optimizer_brute(1e-6, 1, 0.01, nodes, G, generator)
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


def sample(nodelist, parameters, generator):
    samples = []

    for i, nodes in enumerate(nodelist):
        if nodes not in parameters.keys():
            try:
                nodes = np.random.choice(parameters.keys(), 1)
            except:
                nodes = list(parameters.keys())[0]

        if generator == 'BA':
            n = int(parameters[nodes][0])
            m = int(np.rint(parameters[nodes][1]))
            graph = nx.barabasi_albert_graph(n, m)
            while graph.number_of_edges() == 0:
                graph = nx.barabasi_albert_graph(n, m)

        if generator == 'ER':
            n = int(parameters[nodes][0])
            p = parameters[nodes][1]
            graph = nx.fast_gnp_random_graph(n, p)
            while graph.number_of_edges() == 0:
                graph = nx.fast_gnp_random_graph(n, p)

        samples.append(list(graph.edges()))

    return samples


def run_baseline(generator, graphlist):
    # show graphlist statistics
    print('total graph num: {}'.format(len(graphlist)))
    print('max number node: {}'.format(graphlist.max_nodes))

    parameters = train_optimizationbased(graphlist,
                                         generator=generator)

    return parameters
