import networkx as nx


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


def bfs_seq(G, start_id):
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


def encode_graph(G, bfs_order=False):
    if bfs_order:
        start_node = min(G.nodes())
        seq = bfs_seq(G, start_id=start_node)
        # start from 3 because we are also counting pad, sos and eos tokens
        mapping = {n: i for (i, n) in enumerate(seq, 3)}
        G = G.subgraph(seq)
    else:
        # start from 3 because we are also counting pad, sos and eos tokens
        mapping = {n: i for (i, n) in enumerate(G.nodes(), 3)}

    G = nx.relabel_nodes(G, mapping)

    edges = G.edges()
    if bfs_order:
        edges = sorted(edges)

    return list(zip(*edges))


def decode_graph(xs, ys):
    G = nx.Graph()
    G.add_edges_from(zip(xs, ys))
    return G


def decode_graphs(samples):
    graphs = []
    for (xs, ys) in samples:
        graphs.append(decode_graph(xs, ys))
    return GraphList(graphs)
