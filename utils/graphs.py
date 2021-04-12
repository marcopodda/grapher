import networkx as nx


def max_connected_comp(G):
    cc_nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(cc_nodes).copy()