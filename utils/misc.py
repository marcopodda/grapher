import os
import time
import networkx as nx


from .constants import RUNS_DIR, ORDER_DIR
from .serializer import load_yaml


def graph_to_file(G, filename):
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    with open(filename, "w") as f:
        print(f"{G.number_of_nodes()} {G.number_of_edges()}", file=f)
        for e1, e2 in G.edges():
            print(f"{e1} {e2}", file=f)


def maybe_makedir(path):
    if not path.exists():
        os.makedirs(path)


def to_hms(secs):
    return time.strftime('%H:%M:%S', time.gmtime(secs))


def last_in_folder(path):
    return sorted(path.glob("*"))[-1]


def load_result(model_name, dataset_name, order):
    path = RUNS_DIR if order is False else ORDER_DIR
    path = path / model_name / dataset_name
    result_path = last_in_folder(path) / "results" / f"{dataset_name}.yaml"
    return load_yaml(result_path)


def to_latex_row(values, stds=None):
    text = [f"${v:.4f}$" for v in values]
    if stds is not None and not any([x is None for x in stds]):
        text = [f"{v[:-1]} \\pm {std:.4f}$" for (v, std) in zip(text, stds)]
    print(' & '.join(text))


def to_latex_table(all_values, stds_all=None):
    if stds_all is None:
        stds_all = [None] * len(all_values)

    for values, stds, in zip(all_values, stds_all):
        to_latex_row(values, stds)
