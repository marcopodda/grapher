import networkx as nx
import itertools


graphlets = {
    'g1': nx.star_graph(3),
    'g2': nx.connected_caveman_graph(1, 4),
    'g3': nx.cycle_graph(4)
}


def all_matching_subgraphs(G, target):
    for sub_nodes in itertools.combinations(G.nodes(), len(target.nodes())):
        subg = G.subgraph(sub_nodes)
        if nx.is_connected(subg) and nx.is_isomorphic(subg, target):
            yield subg


def graphlet_count(Gs):
    counts = {'g1': [], 'g2': [], 'g3': []}

    for G in Gs:
        count = {'g1': 0, 'g2': 0, 'g3': 0}
        for name, graphlet in graphlets.items():
            for subg in all_matching_subgraphs(G, graphlet):
                count[name] += 1
        for name, value in count.items():
            counts[name].append(value)

    return counts
