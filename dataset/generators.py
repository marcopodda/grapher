import numpy as np
from numpy.random import randint

import io
import zipfile
import requests
import itertools
from pathlib import Path
import networkx as nx


def load_citeseer():
    raw_dir = Path('DATA') / "citeseer" / "raw"

    if not raw_dir.exists():
        url = "http://nrvis.com/download/data/labeled/citeseer.zip"
        response = requests.get(url)
        stream = io.BytesIO(response.content)
        with zipfile.ZipFile(stream) as z:
            for fname in z.namelist():
                z.extract(fname, raw_dir)

    edges_path = raw_dir / "citeseer.edges"
    node_labels_path = raw_dir / "citeseer.node_labels"

    G = nx.Graph()

    with open(edges_path, "r") as f:
        for line in f.readlines():
            n1, n2, _ = line.rsplit("\n")[0].split(",")
            G.add_edge(int(n1) - 1, int(n2) - 1)

    with open(node_labels_path, "r") as f:
        for line in f.readlines():
            node_id, label = line.rsplit("\n")[0].split(",")
            G.nodes[int(node_id) - 1]['label'] = label

    return G


def ego_graph_generator(config, radius=2):
    G = load_citeseer()
    G = max(nx.connected_component_subgraphs(G), key=len)
    G = nx.convert_node_labels_to_integers(G)

    graphs = []
    for i in range(G.number_of_nodes()):
        G_ego = nx.ego_graph(G, i, radius=radius)

        # remove self-loops!
        G_ego.remove_edges_from(G_ego.selfloop_edges())
        num_nodes = G_ego.number_of_nodes()

        if config.min_num_nodes <= num_nodes <= config.max_num_nodes:
            graphs.append(G_ego)

    np.random.shuffle(graphs)
    return graphs


def community_graph_generator2(config, num_graphs=1000, num_communities=2, max_edges=2, intra_connectivity=0.5):
    graphs = []

    for _ in range(num_graphs):
        min_num_nodes, max_num_nodes = config.min_num_nodes, config.max_num_nodes
        n_nodes_communities = [randint(min_num_nodes // num_communities, max_num_nodes // num_communities) for _ in range(num_communities)]
        cumsum_nodes = [sum(n_nodes_communities[:i]) for i in range(len(n_nodes_communities))]

        G = nx.disjoint_union_all([nx.erdos_renyi_graph(n, intra_connectivity) for n in n_nodes_communities])

        for (i, j) in itertools.combinations(range(num_communities), 2):
            for _ in range(randint(1, max_edges + 1)):
                u = randint(cumsum_nodes[i], cumsum_nodes[i] + n_nodes_communities[i])
                v = randint(cumsum_nodes[j], cumsum_nodes[j] + n_nodes_communities[j])
                G.add_edge(u, v)

        G.remove_edges_from(G.selfloop_edges())
        G = max(nx.connected_component_subgraphs(G), key=len)
        G = nx.convert_node_labels_to_integers(G)
        graphs.append(G)

    return graphs

def community_graph_generator(config, num_graphs=1000, num_reps=40, c=2, p_path=0.05, p_edge=0.3):
    graphs = []

    for k in range(config.min_num_nodes, config.max_num_nodes):
        count = 0
        while count < num_reps:
            p = p_path
            path_count = max(int(np.ceil(p * k)), 1)
            G = nx.caveman_graph(c, k)

            # remove 50% edges
            p = 1 - p_edge

            for (u, v) in list(G.edges()):
                if np.random.rand() < p and ((u < k and v < k) or (u >= k and v >= k)):
                    G.remove_edge(u, v)

            # add path_count links
            for i in range(path_count):
                u = np.random.randint(0, k)
                v = np.random.randint(k, k * 2)
                G.add_edge(u, v)

            # remove self-loops!
            G.remove_edges_from(G.selfloop_edges())
            G = max(nx.connected_component_subgraphs(G), key=len)
            G = nx.convert_node_labels_to_integers(G)

            if G.number_of_edges() <= 130 and G.number_of_nodes() >= 10:
                graphs.append(G)

            count += 1

    return graphs


def ladder_graph_generator(config, num_reps):
    min_num_nodes, max_num_nodes = config.min_num_nodes, config.max_num_nodes

    graphs = []

    for num_nodes in range(min_num_nodes // 2, max_num_nodes // 2):
        ladders = [nx.ladder_graph(num_nodes)] * num_reps
        graphs.extend(ladders)

    return graphs
