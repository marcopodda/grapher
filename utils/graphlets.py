import numpy as np
import networkx as nx
import itertools
from joblib import Parallel, delayed


GRAPHLETS = {
    'g1': nx.star_graph(3),
    'g2': nx.star_graph(4),
    'g3': nx.erdos_renyi_graph(4, 1.0),
}


def process_graph(G, nodes, graphlet):
    subg = G.subgraph(nodes)

    if subg.number_of_nodes() != len(nodes):
        return False

    if subg.number_of_edges() != graphlet.number_of_edges():
        return False

    if nx.is_connected(subg) and nx.faster_could_be_isomorphic(subg, graphlet):
        return True

    return False


def _count_star(G, num_nodes):
    neighbors = (itertools.combinations(G.neighbors(n), num_nodes) for n in G.nodes())
    combs = itertools.chain.from_iterable(neighbors)
    return sum(1 for _ in combs)


def count_3star(G):
    return _count_star(G, 3)


def count_4star(G):
    return _count_star(G, 4)


def count_cliques(G):
    count = 0
    for clique in nx.enumerate_all_cliques(G):
        if len(clique) < 4:
            continue
        if len(clique) > 4:
            break
        count += 1
    return count


def average_graphlet_count(graphlist):
    print(len(graphlist))
    counts = Parallel(n_jobs=-1, verbose=1)(delayed(graphlet_count)(G) for G in graphlist)
    counts = np.array(counts).sum(axis=1)
    return counts / counts.sum()


def graphlet_count(G):
    # counts = {'g1': count_3star(G), 'g2': count_4star(G), 'g3': count_cliques(G)}
    # for name, graphlet in GRAPHLETS.items():
    #     num_nodes = graphlet.number_of_nodes() - 1
    #     neighbors = (itertools.combinations([n] + list(G.neighbors(n)), num_nodes) for n in G.nodes())
    #     combs = itertools.chain.from_iterable(neighbors)
    #     results = Parallel(n_jobs=-1)(
    #         delayed(process_graph)(G, nodes, graphlet) for nodes in combs)
    #     counts[name] += sum(results)
    nodes = G.number_of_nodes()
    return [count_3star(G)/nodes, count_4star(G)/nodes, count_cliques(G)/nodes]