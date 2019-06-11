import numpy as np
import io
import zipfile
import requests
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


def ego_graph_generator(min_num_nodes=20, max_num_nodes=40, radius=3):
    G = load_citeseer()
    G = max(nx.connected_component_subgraphs(G), key=len)
    G = nx.convert_node_labels_to_integers(G)

    graphs = []
    for i in range(G.number_of_nodes()):
        G_ego = nx.ego_graph(G, i, radius=radius)
        # remove self-loops!
        G_ego.remove_edges_from(G_ego.selfloop_edges())
        num_nodes = G_ego.number_of_nodes()

        if min_num_nodes <= num_nodes <= max_num_nodes:
            graphs.append(G_ego)

    np.random.shuffle(graphs)
    return graphs


def community_graph_generator(c=2, k=20, p_path=0.05, p_edge=0.3):
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

    return G
