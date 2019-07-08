import networkx as nx
import itertools
from joblib import Parallel, delayed


GRAPHLETS = {
    'g1': nx.star_graph(3),
    'g2': nx.erdos_renyi_graph(4, 1.0),
    'g3': nx.cycle_graph(4)
}


def process_graph(G, nodes, graphlet):
    subg = G.subgraph(nodes)
    if nx.is_connected(subg) and nx.faster_could_be_isomorphic(subg, graphlet):
        return True
    return False


def graphlet_count(G):
    counts = {}
    for name, graphlet in GRAPHLETS.items():
        num_nodes = graphlet.number_of_nodes()
        combs = itertools.combinations(G.nodes(), num_nodes)
        results = Parallel(n_jobs=-1)(
            delayed(process_graph)(G, nodes, graphlet) for nodes in combs)
        counts[name] = sum(results)
    return counts