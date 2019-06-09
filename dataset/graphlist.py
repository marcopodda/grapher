class GraphList:
    """
    A class for a list of networkx graphs.
    """

    def __init__(self, graphs=[]):
        self._graphs = graphs
        self._max_nodes = None
        self._max_edges = None
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

    @property
    def max_nodes(self):
        if self._max_nodes is None:
            num_nodes = [G.number_of_nodes() for G in self]
            self._max_nodes = max(num_nodes)
        return self._max_nodes

    @property
    def max_edges(self):
        if self._max_edges is None:
            num_edges = [G.number_of_edges() for G in self]
            self._max_edges = max(num_edges)
        return self._max_edges

    @property
    def avg_nodes(self):
        if self._avg_nodes is None:
            num_nodes = [G.number_of_nodes() for G in self]
            self._avg_nodes = sum(num_nodes) / len(self)
        return self._avg_nodes

    @property
    def avg_edges(self):
        if self._avg_edges is None:
            num_edges = [G.number_of_edges() for G in self]
            self._avg_edges = sum(num_edges) / len(self)
        return self._avg_edges

    def filter(self, fn):
        self._graphs = [el for el in self._graphs if fn(el)]
        self._max_nodes = None
        self._max_edges = None
        self._avg_nodes = None
        self._avg_edges = None