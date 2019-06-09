from pathlib import Path

import numpy as np
import networkx as nx


def read_data(dataset_name, root_dir, use_node_labels=True):
    node2graph = {}
    Gs = []

    graph_indicator_path = root_dir / dataset_name / f"{dataset_name}_graph_indicator.txt"
    adj_list_path = root_dir / dataset_name / f"{dataset_name}_A.txt"
    node_labels_path = root_dir / dataset_name / f"{dataset_name}_node_labels.txt"
    graph_labels_path = root_dir / dataset_name / f"{dataset_name}_graph_labels.txt"


    with open(graph_indicator_path, "r") as f:
        c = 1
        for line in f:
            node2graph[c] = int(line[:-1])
            if not node2graph[c] == len(Gs):
                Gs.append(nx.Graph())
            Gs[-1].add_node(c)
            c += 1

    with open(adj_list_path, "r") as f:
        for line in f:
            edge = line[:-1].split(",")
            edge[1] = edge[1].replace(" ", "")
            Gs[node2graph[int(edge[0])]-1].add_edge(int(edge[0]), int(edge[1]))

    if use_node_labels:
        with open(node_labels_path, "r") as f:
            c = 1
            for line in f:
                node_label = int(line[:-1])
                Gs[node2graph[c]-1].node[c]['label'] = node_label
                c += 1

    labels = []
    with open(graph_labels_path, "r") as f:
        for line in f:
            labels.append(int(line[:-1]))

    labels  = np.array(labels, dtype = np.float)
    return Gs, labels


