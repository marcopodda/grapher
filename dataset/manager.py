import os
import io
import requests
import zipfile
from pathlib import Path
import operator
import numpy as np

import networkx as nx
from nltk.tree import ParentedTree
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from .generators import community_graph_generator, ego_graph_generator, ladder_graph_generator
from .graph import GraphList
from .dataset import GraphDataset, GraphDataCollator
from utils.serializer import load_yaml, save_yaml
from config import get_config_class


DATA_DIR = Path('DATA')


class DatasetManager:
    def __init__(self, config, exp_root, name):
        self.name = name
        self.exp_root = exp_root
        self.config = config

        self.raw_dir = DATA_DIR / name / "raw"
        if not self.raw_dir.exists():
            os.makedirs(self.raw_dir)
            self._fetch_data()

        self.processed_dir = exp_root / "data"
        if not self.processed_dir.exists():
            os.makedirs(self.processed_dir)

        if not (self.raw_dir / f"{self.name}.pt").exists():
            self._preprocess_data()

        self.data = torch.load(self.raw_dir / f"{self.name}.pt")
        # save a copy inside the experiment folder too
        torch.save(self.data, self.processed_dir / f"{self.name}.pt")

        if not (self.raw_dir / f"splits.yaml").exists():
            self._make_splits()

        self.splits = load_yaml(self.raw_dir / f"splits.yaml")
        # save a copy inside the experiment folder too
        save_yaml(self.splits, self.processed_dir / 'splits.yaml')

    def _preprocess_data(self):
        graphlist = self._read_data()
        dataset = GraphDataset(self.config, graphlist)
        torch.save(dataset, self.raw_dir / f"{self.name}.pt")

    def _make_splits(self):
        indices = [i for i in range(len(self.data))]
        train_idxs, test_idxs = train_test_split(indices, test_size=self.config.test_size)
        splits = {'train': train_idxs, 'test': test_idxs}
        save_yaml(splits, self.raw_dir / 'splits.yaml')

    def get_loader(self, name):
        indices = self.splits[name]
        return DataLoader(dataset=Subset(self.data, indices),
                          batch_size=self.config.batch_size,
                          shuffle=self.config.shuffle,
                          collate_fn=GraphDataCollator(self.config))

    def get_data(self, name):
        indices = self.splits[name]
        graphs = operator.itemgetter(*indices)(self.data.graphlist)
        return GraphList(graphs)

    def __len__(self):
        return len(self.data)

    @property
    def input_dim(self):
        return self.data.input_dim

    @property
    def output_dim(self):
        return self.data.output_dim


class TUData(DatasetManager):
    URL = "http://ls11-www.cs.uni-dortmund.de/people/morris/graphkerneldatasets/{name}.zip"

    def _fetch_data(self):
        url = self.URL.format(name=self.name)
        response = requests.get(url)
        stream = io.BytesIO(response.content)
        with zipfile.ZipFile(stream) as z:
            for fname in z.namelist():
                z.extract(fname, self.raw_dir)

    def _read_data(self):
        node2graph = {}
        Gs = []

        graph_indicator_path = self.raw_dir / self.name / f"{self.name}_graph_indicator.txt"
        adj_list_path = self.raw_dir / self.name / f"{self.name}_A.txt"
        node_labels_path = self.raw_dir / self.name / f"{self.name}_node_labels.txt"

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

        if node_labels_path.exists():
            with open(node_labels_path, "r") as f:
                c = 1
                for line in f:
                    node_label = int(line[:-1])
                    Gs[node2graph[c]-1].nodes[c]['label'] = node_label
                    c += 1

        graphlist = GraphList(Gs)

        graphlist = graphlist.filter(lambda G: G.number_of_nodes() <= self.config.max_num_nodes)
        graphlist = graphlist.filter(lambda G: G.number_of_nodes() >= self.config.min_num_nodes)
        graphlist = graphlist.filter(lambda G: G.number_of_edges() <= self.config.max_num_edges)
        graphlist = graphlist.filter(lambda G: G.number_of_edges() >= self.config.min_num_edges)

        return graphlist


class Trees(DatasetManager):
    def _fetch_data(self):
        pass

    def _make_splits(self):
        splits = {'train': list(range(600)), 'test': list(range(600, 780))}
        save_yaml(splits, self.raw_dir / 'splits.yaml')

    def _read_data(self):
        def set_index(tree):
            tree.set_label(str(id(tree)))

            for st in tree:
                set_index(st)


        def build_graph(G, tree):
            for st in tree:
                build_graph(G, st)

            if tree.parent():
                G.add_edge(tree.label(), tree.parent().label())

        graphs = []
        with open(self.raw_dir / "tr.txt", "r") as f:
            for line in f.readlines():
                tree_string = line.rstrip("\n")
                tree = ParentedTree.fromstring(tree_string, brackets="{}")
                set_index(tree)

                G = nx.Graph()
                build_graph(G, tree)
                graphs.append(G)

        with open(self.raw_dir / "test.txt", "r") as f:
            for line in f.readlines():
                tree_string = line.rstrip("\n")
                tree = ParentedTree.fromstring(tree_string, brackets="{}")
                set_index(tree)

                G = nx.Graph()
                build_graph(G, tree)
                graphs.append(G)

        return GraphList(graphs)


class SyntheticData(DatasetManager):

    def __init__(self, config, exp_root, name):
        super().__init__(config, exp_root, name)

        if 'generator_kwargs' in config:
            self.generator_kwargs.update(**config.generator_kwargs)

    def _fetch_data(self):
        pass

    def _read_data(self):
        generator = self.get_generator()
        graphs = generator(self.config, **self.generator_kwargs)
        graphlist = GraphList(graphs)

        graphlist = graphlist.filter(lambda G: G.number_of_nodes() <= self.config.max_num_nodes)
        graphlist = graphlist.filter(lambda G: G.number_of_nodes() >= self.config.min_num_nodes)
        graphlist = graphlist.filter(lambda G: G.number_of_edges() <= self.config.max_num_edges)
        graphlist = graphlist.filter(lambda G: G.number_of_edges() >= self.config.min_num_edges)

        return graphlist

    def get_generator(self):
        raise NotImplementedError


class Community(SyntheticData):
    generator_kwargs = {
        "num_graphs": 1000,
    }

    def get_generator(self):
        return community_graph_generator


class Ego(SyntheticData):
    generator_kwargs = {
        "radius": 2
    }

    def get_generator(self):
        return ego_graph_generator


class Ladders(SyntheticData):
    max_iters = 3000
    generator_kwargs = {
        "num_reps": 10
    }

    def get_generator(self):
        return ladder_graph_generator

    def _make_splits(self):
        num_nodes = [G.number_of_nodes() for G in self.data.graphlist]
        test_size = len(self.data) // self.generator_kwargs['num_reps']
        indices = [i for i in range(len(self.data))]
        train_idxs, test_idxs = train_test_split(indices, stratify=num_nodes, test_size=test_size)
        splits = {'train': train_idxs, 'test': test_idxs}
        save_yaml(splits, self.raw_dir / 'splits.yaml')
        # save a copy inside the experiment folder too
        save_yaml(splits, self.processed_dir / 'splits.yaml')


def get_dataset_class(name):
    if name.lower() == 'community':
        return Community

    if name.lower() == 'trees':
        return Trees

    if name.lower() == 'ego':
        return Ego

    if name.lower() == 'ladders':
        return Ladders

    return TUData


def load_dataset(dataset_name, model_name, exp):
    config_class = get_config_class(model_name)
    config = config_class.from_file(exp.root / "config" / "config.yaml")
    dataset_class = get_dataset_class(dataset_name)
    dataset = dataset_class(config, exp.root, dataset_name)
    return dataset