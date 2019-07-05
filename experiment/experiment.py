import os
import numpy as np
from datetime import datetime
from pathlib import Path

import torch

from baselines.simple import run_baseline, sample as sample_baseline
from baselines.graphrnn.run import run_graphrnn, load_model
from baselines.graphrnn.train import sample as sample_graphrnn
from baselines.gru.trainer import GRUTrainer
from baselines.gru.data import GRUDataset, GRUDataCollator, build_vocab
from config.config import Config, BaselineConfig, GraphRNNConfig, GRUConfig

from dataset import get_dataset_class
from dataset.graph import GraphList
from learner.trainer import Trainer
from utils.training import get_device
from utils.evaluation import filter_unique_and_novel, clean_graph


RUNS_DIR = Path('RUNS')


def maybe_makedir(path):
    if not path.exists():
        os.makedirs(path)


class BaseExperiment:
    @classmethod
    def load(cls, root):
        assert root is not None
        dataset = Path(root).parts[-2]
        return cls(dataset, root=root)

    def __init__(self, dataset, root=None):
        self.dataset = dataset
        self.dataset_class = get_dataset_class(dataset)

        if root is None:
            now = datetime.now().isoformat()
            self.name = f"{self.dataset}_{now}"
            self.root = RUNS_DIR / f"{self.model_name}" / f"{self.dataset}" / f"{now}"
        else:
            self.root = Path(root)
            self.name = "_".join(self.root.parts[-2:])

        maybe_makedir(self.root)
        maybe_makedir(self.root / "ckpt")
        maybe_makedir(self.root / "data")
        maybe_makedir(self.root / "config")
        maybe_makedir(self.root / "samples")
        maybe_makedir(self.root / "results")

    def train(self):
        raise NotImplementedError

    def sample(self, num_samples):
        raise NotImplementedError

    def sample_novel_and_unique(self, num_samples):
        raise NotImplementedError


class Experiment(BaseExperiment):
    model_name = "GRAPHER"

    def train(self):
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
        return GraphList([clean_graph(e) for e in samples])

    def sample_novel_and_unique(self, num_samples):
        config = Config.from_file(self.root / "config" / f"config.yaml")
        dataset = self.dataset_class(config, self.root, name=self.dataset)
        trainer = Trainer.load(config, self.root, dataset.input_dim, dataset.output_dim, best=True)

        samples = []
        train_data = dataset.get_data('train')
        while len(samples) < num_samples:
            sample = trainer.sample(num_samples=num_samples)
            sample = filter_unique_and_novel(train_data, [clean_graph(e) for e in samples])
            samples.extend(sample)

        return GraphList(samples[:num_samples])


class OrderExperiment(Experiment):
    model_name = "ORDER"

    @classmethod
    def load(cls, root):
        assert root is not None
        dataset = Path(root).parts[-2]
        order_type = Path(root).parts[-3]
        return cls(order_type, dataset, root=root)

    def __init__(self, order_type, dataset, root=None):
        super().__init__(dataset, root=root)
        self.order_type = order_type

    def train(self):
        config = Config.from_file(Path("cfg") / f"config_{self.dataset}.yaml")
        config.update(order=self.order_type)
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
        e2i, i2e = build_vocab(dataset.data.graphlist)
        input_dim = output_dim = len(e2i)

        loader = torch.utils.data.DataLoader(
            GRUDataset(config, e2i, train_data),
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            collate_fn=GRUDataCollator(config))

        trainer = GRUTrainer(config, self.root, input_dim, output_dim, i2e)
        trainer.fit(loader)
        config.save(self.root / "config")

    def sample(self, num_samples):
        config = GRUConfig.from_file(self.root / "config" / f"config.yaml")
        dataset = self.dataset_class(config, self.root, name=self.dataset)
        e2i, i2e = build_vocab(dataset.data.graphlist)
        input_dim = output_dim = len(e2i)
        trainer = GRUTrainer.load(config, self.root, input_dim, output_dim, i2e, best=True)
        samples = trainer.sample(num_samples=num_samples)
        samples = [[i2e[i] for i in sample] for sample in samples]
        return GraphList(samples)

    def sample_novel_and_unique(self, num_samples):
        config = GRUConfig.from_file(self.root / "config" / f"config.yaml")
        dataset = self.dataset_class(config, self.root, name=self.dataset)
        e2i, i2e = build_vocab(dataset.data.graphlist)
        input_dim = output_dim = len(e2i)
        trainer = GRUTrainer.load(config, self.root, input_dim, output_dim, i2e, best=True)

        samples = []
        train_data = dataset.get_data('train')
        while len(samples) < num_samples:
            sample = trainer.sample(num_samples=num_samples)
            sample = [[i2e[i] for i in sample] for sample in samples]
            sample = filter_unique_and_novel(train_data, [clean_graph(e) for e in samples])
            samples.extend(sample)

        return GraphList(samples[:num_samples])


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
        device = get_device(config)
        rnn_state_dict = torch.load(self.root / "ckpt" / f"rnn.pt", map_location=device)
        output_state_dict = torch.load(self.root / "ckpt" / f"output.pt", map_location=device)
        rnn, output = load_model(config, rnn_state_dict, output_state_dict)
        samples = sample_graphrnn(config, rnn, output, num_samples=num_samples)
        return GraphList([clean_graph(e) for e in samples])

    def sample_novel_and_unique(self, num_samples):
        config = GraphRNNConfig.from_file(self.root / "config" / f"config.yaml")
        device = get_device(config)
        dataset = self.dataset_class(config, self.root, name=self.dataset)
        rnn_state_dict = torch.load(self.root / "ckpt" / f"rnn.pt", map_location=device)
        output_state_dict = torch.load(self.root / "ckpt" / f"output.pt", map_location=device)
        rnn, output = load_model(config, rnn_state_dict, output_state_dict)
        samples = sample_graphrnn(config, rnn, output, num_samples=num_samples)

        samples = []
        train_data = dataset.get_data('train')
        while len(samples) < num_samples:
            sample = sample_graphrnn(config, rnn, output, num_samples=num_samples)
            sample = filter_unique_and_novel(train_data, [clean_graph(e) for e in samples])
            samples.extend(sample)

        return GraphList(samples[:num_samples])


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
        return samples

    def sample_novel_and_unique(self, num_samples):
        config = BaselineConfig.from_file(self.root / "config" / f"config.yaml")
        dataset = self.dataset_class(config, self.root, name=self.dataset)
        parameters = torch.load(self.root / "ckpt" / f"parameters.pt")

        samples = []
        train_data = dataset.get_data('train')
        while len(samples) < num_samples:
            nodes = [G.number_of_nodes() for G in dataset.get_data('test')]
            nodes = np.random.choice(nodes, num_samples)
            sample = sample_baseline(nodes, parameters=parameters, generator=self.model_name)
            sample = filter_unique_and_novel(train_data, [clean_graph(e) for e in samples])
            samples.extend(sample)

        return GraphList(samples[:num_samples])


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
    rundir = root / model_name / dataset_name
    expdir = sorted(rundir.glob("*"))
    assert len(expdir) > 0
    exp_class = get_exp_class(model_name)
    exp = exp_class.load(expdir[-1])
    return exp