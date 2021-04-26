import os
import numpy as np
from datetime import datetime
from pathlib import Path

import torch

from baselines.simple import run_baseline, sample as sample_baseline
from baselines.graphrnn.run import run_graphrnn, load_model
from baselines.graphrnn.train import sample as sample_graphrnn
from baselines.gru.trainer import GRUTrainer
from baselines.gru.data import GRUDataset, GRUDataCollator
from config.config import Config, BaselineConfig, GraphRNNConfig, GRUConfig

from dataset import get_dataset_class
from dataset.graph import GraphList
from learner.trainer import Trainer
from utils.training import get_device
from utils.constants import RUNS_DIR
from utils.evaluation import clean_graph
from utils.misc import last_in_folder, maybe_makedir



class BaseExperiment:
    _root = RUNS_DIR

    @classmethod
    def load(cls, root):
        assert root is not None
        dataset = Path(root).parts[-1]
        return cls(dataset, root=root, exist_ok=True)

    def __init__(self, dataset, root=None, exist_ok=False):
        self.dataset = dataset
        self.dataset_class = get_dataset_class(dataset)

        if root is None:
            self.name = f"{self.dataset}"
            self.root = self._root / f"{self.model_name}" / f"{self.dataset}"
        else:
            self.root = Path(root)
            self.name = "_".join(self.root.parts[-2:])

        self.root.mkdir(parents=True, exist_ok=exist_ok)
        (self.root / "ckpt").mkdir(exist_ok=exist_ok)
        (self.root / "data").mkdir(exist_ok=exist_ok)
        (self.root / "config").mkdir(exist_ok=exist_ok)
        (self.root / "samples").mkdir(exist_ok=exist_ok)
        (self.root / "results").mkdir(exist_ok=exist_ok)

    def train(self):
        raise NotImplementedError

    def sample(self, num_samples):
        raise NotImplementedError


class Experiment(BaseExperiment):
    model_name = "GRAPHER"

    def train(self):
        print(f"Training on {self.dataset} on order dfs-fixed")
        config = Config.from_file(Path("cfg") / f"config_{self.dataset}.yaml")
        dataset = self.dataset_class(config, self.root, name=self.dataset)
        trainer = Trainer(config, self.root, dataset.input_dim, dataset.output_dim)
        loader = dataset.get_loader('train')
        trainer.fit(loader, order=config.order)
        config.save(self.root / "config")

    def sample(self, num_samples):
        config = Config.from_file(self.root / "config" / f"config.yaml")
        dataset = self.dataset_class(config, self.root, name=self.dataset)
        trainer = Trainer.load(config, self.root, dataset.input_dim, dataset.output_dim, best=True)
        samples = trainer.sample(num_samples=num_samples)
        return [clean_graph(e) for e in samples]


class OrderExperiment(Experiment):
    _root = RUNS_DIR / "ORDER"

    @classmethod
    def load(cls, root):
        dataset = Path(root).parts[-1]
        order_type = Path(root).parts[-2]
        return cls(order_type, dataset, root=root, exist_ok=True)

    def __init__(self, order_type, dataset, root=None, exist_ok=False):
        self.model_name = order_type
        super().__init__(dataset, root=root, exist_ok=exist_ok)

    def train(self):
        print(f"Training on {self.dataset} on order {self.model_name}")
        config = Config.from_file(Path("cfg") / f"config_{self.dataset}.yaml")
        config.update(order=self.model_name)
        dataset = self.dataset_class(config, self.root, name=self.dataset)
        trainer = Trainer(config, self.root, dataset.input_dim, dataset.output_dim)
        loader = dataset.get_loader('train')
        trainer.fit(loader, order=config.order)
        config.save(self.root / "config")


class GRUExperiment(BaseExperiment):
    model_name = "GRU"

    def train(self):
        config = GRUConfig.from_file(Path("cfg") / f"GRU_{self.dataset}.yaml")
        dataset = self.dataset_class(config, self.root, name=self.dataset)
        train_data = dataset.get_data('train')

        loader = torch.utils.data.DataLoader(
            GRUDataset(config, train_data),
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            collate_fn=GRUDataCollator(config))

        trainer = GRUTrainer(config, self.root, dataset.input_dim, dataset.output_dim)
        trainer.fit(loader)
        config.save(self.root / "config")

    def sample(self, num_samples):
        config = GRUConfig.from_file(self.root / "config" / f"config.yaml")
        dataset = self.dataset_class(config, self.root, name=self.dataset)
        trainer = GRUTrainer.load(config, self.root, dataset.input_dim, dataset.output_dim, best=True)
        samples = trainer.sample(num_samples=num_samples)
        return GraphList([clean_graph(e) for e in samples])


class GraphRNNExperiment(BaseExperiment):
    model_name = "GRAPHRNN"

    def train(self):
        config = GraphRNNConfig.from_file(Path("cfg") / f"graphrnn_{self.dataset}.yaml")
        dataset = self.dataset_class(config, self.root, name=self.dataset)
        train_data = dataset.get_data('train')
        run_graphrnn(config, self.dataset, self.root, train_data)
        config.save(self.root / "config")

    def sample(self, num_samples):
        config = GraphRNNConfig.from_file(self.root / "config" / f"config.yaml")
        device = get_device("cpu")
        rnn_state_dict = torch.load(self.root / "ckpt" / f"rnn.pt", map_location=device)
        output_state_dict = torch.load(self.root / "ckpt" / f"output.pt", map_location=device)
        rnn, output = load_model(config, rnn_state_dict, output_state_dict)
        samples = sample_graphrnn(config, rnn, output, num_samples=num_samples)
        return GraphList([clean_graph(e) for e in samples])


class BaselineExperiment(BaseExperiment):

    def train(self):
        config = BaselineConfig.from_file(Path("cfg") / f"baseline_{self.dataset}.yaml")
        config.update(name=self.model_name)
        config.save(self.root / "config")

        dataset = self.dataset_class(config, self.root, name=self.dataset)

        train_data = dataset.get_data('train')
        parameters = run_baseline(self.model_name, train_data)
        torch.save(parameters, self.root / "ckpt" / f"parameters.pt")

    def sample(self, num_samples):
        config = BaselineConfig.from_file(self.root / "config" / f"config.yaml")
        dataset = self.dataset_class(config, self.root, name=self.dataset)
        parameters = torch.load(self.root / "ckpt" / f"parameters.pt")
        nodes = [G.number_of_nodes() for G in dataset.get_data('test')]
        nodes = np.random.choice(nodes, num_samples)
        samples = sample_baseline(nodes, parameters=parameters, generator=self.model_name)
        return GraphList([clean_graph(e) for e in samples])


class ERExperiment(BaselineExperiment):
    model_name = "ER"


class BAExperiment(BaselineExperiment):
    model_name = "BA"


def get_exp_class(model_name):
    if model_name == "GRAPHER":
        return Experiment

    if model_name == "GRAPHRNN":
        return GraphRNNExperiment

    if model_name == "ER":
        return ERExperiment

    if model_name == "BA":
        return BAExperiment

    if model_name == "GRU":
        return GRUExperiment

    return OrderExperiment


def load_experiment(root, model_name, dataset_name):
    expdir = root / model_name / dataset_name
    exp_class = get_exp_class(model_name)
    exp = exp_class.load(expdir)
    return exp
