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


def graphlet_count(G):
    counts = {'g1': 0, 'g2': 0, 'g3': 0}
    for name, graphlet in GRAPHLETS.items():
        num_nodes = graphlet.number_of_nodes()
        neighbors = (itertools.combinations([n] + list(G.neighbors(n)), num_nodes) for n in G.nodes())
        combs = itertools.chain.from_iterable(neighbors)
        results = Parallel(n_jobs=-1)(
            delayed(process_graph)(G, nodes, graphlet) for nodes in combs)
        counts[name] += sum(results)
    return counts