import os
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
from .trainer import Trainer
from utils.training import get_device


RUNS_DIR = Path('RUNS')


def maybe_makedir(path):
    if not path.exists():
        os.makedirs(path)


class Experiment:
    model_name = "GRAPHER"

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
            self.root = RUNS_DIR / self.model_name / f"{self.dataset}" / f"{now}"
        else:
            self.root = Path(root)
            self.name = "_".join(self.root.parts[-2:])

        maybe_makedir(self.root)
        maybe_makedir(self.root / "ckpt")
        maybe_makedir(self.root / "data")
        maybe_makedir(self.root / "config")
        maybe_makedir(self.root / "samples")

    def train(self):
        config = Config.from_file(Path("cfg") / f"config_{self.dataset}.yaml")
        dataset = self.dataset_class(config, self.root, name=self.dataset)
        trainer = Trainer(config, self.root, dataset.input_dim, dataset.output_dim)
        loader = dataset.get_loader('train')
        trainer.fit(loader)
        config.save(self.root / "config")

    def sample(self, num_samples):
        config = Config.from_file(self.root / "config" / f"config.yaml")
        dataset = self.dataset_class(config, self.root, name=self.dataset)
        trainer = Trainer.load(config, self.root, dataset.input_dim, dataset.output_dim, best=True)
        samples, iters, duplicate_train, duplicate_sample = trainer.sample(
            dataset.get_data('train'), dataset.max_iters, num_samples=num_samples)
        return samples, iters, duplicate_train, duplicate_sample


class GRUExperiment:
    model_name = "GRU"

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
            self.root = RUNS_DIR / self.model_name / f"{self.dataset}" / f"{now}"
        else:
            self.root = Path(root)
            self.name = "_".join(self.root.parts[-2:])

        maybe_makedir(self.root)
        maybe_makedir(self.root / "ckpt")
        maybe_makedir(self.root / "data")
        maybe_makedir(self.root / "config")
        maybe_makedir(self.root / "samples")

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
        samples, iters, duplicate_train, duplicate_sample = trainer.sample(
            dataset.get_data('train'), i2e, dataset.max_iters, num_samples=num_samples)
        return samples, iters, duplicate_train, duplicate_sample


class BaselineExperiment(Experiment):
    @classmethod
    def load(cls, root):
        assert root is not None
        dataset = Path(root).parts[-2]
        return cls(dataset, root=root)

    def __init__(self, dataset, root=None):
        self.dataset = dataset
        self.model_name, self.metric = self.model_name.split("_")
        self.dataset_class = get_dataset_class(dataset)

        if root is None:
            now = datetime.now().isoformat()
            self.name = f"{self.dataset}_{now}"
            self.root = RUNS_DIR / f"{self.model_name}_{self.metric}" / f"{self.dataset}" / f"{now}"
        else:
            self.root = Path(root)
            self.name = "_".join(self.root.parts[-2:])

        maybe_makedir(self.root)
        maybe_makedir(self.root / "ckpt")
        maybe_makedir(self.root / "data")
        maybe_makedir(self.root / "config")
        maybe_makedir(self.root / "samples")

    def train(self):
        config = BaselineConfig.from_file(Path("cfg") / f"baseline_{self.dataset}.yaml")
        config.update(metric=self.metric, name=self.model_name)
        config.save(self.root / "config")

        dataset = self.dataset_class(config, self.root, name=self.dataset)

        train_data = dataset.get_data('train')
        parameters = run_baseline(self.model_name, self.metric, train_data)
        torch.save(parameters, self.root / "ckpt" / f"parameters.pt")

    def sample(self, num_samples):
        config = BaselineConfig.from_file(self.root / "config" / f"config.yaml")
        dataset = self.dataset_class(config, self.root, name=self.dataset)
        parameters = torch.load(self.root / "ckpt" / f"parameters.pt")
        samples = sample_baseline(dataset.get_data('test'), parameters=parameters, generator=self.model_name)
        return samples


class ERDegreeExperiment(BaselineExperiment):
    model_name = "ER_degree"


class ERClusteringExperiment(BaselineExperiment):
    model_name = "ER_clustering"


class BADegreeExperiment(BaselineExperiment):
    model_name = "BA_degree"


class BAClusteringExperiment(BaselineExperiment):
    model_name = "BA_clustering"


class GraphRNNExperiment(Experiment):
    model_name = "GRAPHRNN"

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

    def train(self):
        config = GraphRNNConfig.from_file(Path("cfg") / f"graphrnn_{self.dataset}.yaml")
        dataset = self.dataset_class(config, self.root, name=self.dataset)
        train_data = dataset.get_data('train')
        run_graphrnn(config, self.dataset, self.root, train_data)
        config.save(self.root / "config")

    def sample(self, num_samples):
        config = GraphRNNConfig.from_file(self.root / "config" / f"config.yaml")
        device = get_device(config)
        dataset = self.dataset_class(config, self.root, name=self.dataset)
        rnn_state_dict = torch.load(self.root / "ckpt" / f"rnn.pt", map_location=device)
        output_state_dict = torch.load(self.root / "ckpt" / f"output.pt", map_location=device)
        rnn, output = load_model(config, rnn_state_dict, output_state_dict)
        samples, iters, duplicate_train, duplicate_sample = sample_graphrnn(
            config, rnn, output, dataset.get_data('train'), dataset.max_iters, num_samples=num_samples)
        return samples, iters, duplicate_train, duplicate_sample


class OrderExperiment(Experiment):
    model_name = "ORDER"

    @classmethod
    def load(cls, root):
        assert root is not None
        dataset = Path(root).parts[-2]
        order_type = Path(root).parts[-3]
        return cls(order_type, dataset, root=root)

    def __init__(self, order_type, dataset, root=None):
        self.dataset = dataset
        self.dataset_class = get_dataset_class(dataset)
        self.order_type = order_type

        if root is None:
            now = datetime.now().isoformat()
            self.name = f"{self.dataset}_{now}"
            self.root = RUNS_DIR / self.model_name / self.order_type / f"{self.dataset}" / f"{now}"
        else:
            self.root = Path(root)
            self.name = "_".join(self.root.parts[-2:])

        maybe_makedir(self.root)
        maybe_makedir(self.root / "ckpt")
        maybe_makedir(self.root / "data")
        maybe_makedir(self.root / "config")
        maybe_makedir(self.root / "samples")

    def train(self):
        config = Config.from_file(Path("cfg") / f"config_{self.dataset}.yaml")
        config.update(order=self.order_type)
        dataset = self.dataset_class(config, self.root, name=self.dataset)
        trainer = Trainer(config, self.root, dataset.input_dim, dataset.output_dim)
        loader = dataset.get_loader('train')
        test_data = dataset.get_data('test')
        trainer.fit(loader, test_data)
        config.save(self.root / "config")
