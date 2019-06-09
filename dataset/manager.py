import os
import io
import requests
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from .utils import read_data
from .graphlist import GraphList
from .dataset import GraphDataset


DATA_DIR = Path('DATA')
URL = "http://ls11-www.cs.uni-dortmund.de/people/morris/graphkerneldatasets/{name}.zip"

class DatasetManager:

    def __init__(self, name, config):
        self.name = name
        self.config = config

        self.raw_dir = DATA_DIR / name / "raw"
        if not self.raw_dir.exists():
            os.makedirs(self.raw_dir)
            self._download()

        self.processed_dir = DATA_DIR / name / "processed"
        if not self.processed_dir.exists():
            os.makedirs(self.processed_dir)
            self._preprocess()

        self.data = torch.load(self.processed_dir / f"{self.name}.pt")

        splits_file = self.processed_dir / 'splits.pt'
        if not splits_file.exists():
            self._make_splits()

        self.splits = torch.load(self.processed_dir / 'splits.pt')


    def _download(self):
        url = URL.format(name=self.name)
        response = requests.get(url)
        stream = io.BytesIO(response.content)
        with zipfile.ZipFile(stream) as z:
            for fname in z.namelist():
                z.extract(fname, self.raw_dir)

    def _preprocess(self):
        graphs, _ = read_data(self.name, self.raw_dir)
        graphlist = GraphList(graphs)

        if self.config.max_num_nodes:
            graphlist.filter(lambda el: el.number_of_nodes() < self.config.max_num_nodes)

        if self.config.max_num_edges:
            graphlist.filter(lambda el: el.number_of_edges() < self.config.max_num_edges)

        torch.save(GraphDataset(graphlist), self.processed_dir / f"{self.name}.pt")

    def _make_splits(self):
        indices = [i for i in range(len(self.data))]
        train_idxs, test_idxs = train_test_split(indices, test_size=self.config.test_size)
        splits = {'train': train_idxs, 'test': test_idxs}
        torch.save(splits, self.processed_dir / 'splits.pt')

    def get_loader(self, name):
        indices = self.splits[name]
        dataset = Subset(self.data, indices)
        return DataLoader(dataset,
                          batch_size=self.config.batch_size,
                          shuffle=self.config.shuffle)


