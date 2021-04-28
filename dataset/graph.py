import numpy as np
import networkx as nx

from utils.graphs import max_connected_comp


class GraphList:
    """
    A class for a list of networkx graphs.
    """

    def __init__(self, graphs=[]):
        self._graphs = graphs
        self._max_nodes = None
        self._max_edges = None
        self._min_nodes = None
        self._min_edges = None
        self._avg_nodes = None
        self._avg_edges = None

    def __iter__(self):
        return iter(self._graphs)

    def __len__(self):
        return len(self._graphs)

    def __getitem__(self, index):
        return self._graphs[index]

    def __setitem__(self, index, item):
        self._graphs[index] = item

    def __contains__(self, item):
        return item in self._graphs

    def num_nodes(self):
        return [G.number_of_nodes() for G in self]

    def nodes(self):
        return [list(G.nodes()) for G in self]

    @property
    def max_nodes(self):
        if self._max_nodes is None:
            self._max_nodes = max(self.num_nodes())
        return self._max_nodes

    @property
    def max_edges(self):
        if self._max_edges is None:
            num_edges = [G.number_of_edges() for G in self]
            self._max_edges = max(num_edges)
        return self._max_edges

    @property
    def min_nodes(self):
        if self._min_nodes is None:
            self._min_nodes = min(self.num_nodes())
        return self._min_nodes

    @property
    def min_edges(self):
        if self._min_edges is None:
            num_edges = [G.number_of_edges() for G in self]
            self._min_edges = min(num_edges)
        return self._min_edges

    @property
    def avg_nodes(self):
        if self._avg_nodes is None:
            self._avg_nodes = sum(self.num_nodes()) / len(self)
        return self._avg_nodes

    @property
    def avg_edges(self):
        if self._avg_edges is None:
            num_edges = [G.number_of_edges() for G in self]
            self._avg_edges = sum(num_edges) / len(self)
        return self._avg_edges

    def filter(self, fn):
        graphs = [el for el in self._graphs if fn(el)]
        return GraphList(graphs)


def dfs_order(G, start_id):
    dictionary = dict(nx.dfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                next = next + neighbor
        output = output + next
        start = next
    return output


def bfs_order(G, start_id):
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                next = next + neighbor
        output = output + next
        start = next
    return output


def encode_graph(G, order):
    G = max_connected_comp(G)
    mapping = {n: i for (i, n) in enumerate(G.nodes(), 3)}
    G = nx.relabel_nodes(G, mapping)

    nodes_list = list(G.nodes())

    if order == "bfs-fixed":
        start_node = min(nodes_list)
        seq = bfs_order(G, start_id=start_node)
    elif order == "bfs-random":
        start_node = np.random.choice(nodes_list)
        seq = bfs_order(G, start_id=start_node)
    elif order == "dfs-random":
        start_node = np.random.choice(nodes_list)
        seq = dfs_order(G, start_id=start_node)
    elif order == "dfs-fixed":
        start_node = min(nodes_list)
        seq = dfs_order(G, start_id=start_node)
    elif order == "random":
        seq = nodes_list
        np.random.shuffle(seq)
    elif order == "smiles":
        # this is the case of SMILES ordering, which
        # for chemical datasets is given by default
        seq = nodes_list
    else:
        raise ValueError

    edges = G.edges()
    if order != "smiles":
        edges = sorted(edges)

    return list(zip(*edges))
