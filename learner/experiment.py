import os
from datetime import datetime
from pathlib import Path

import torch

from baselines.simple import run_baseline, sample as sample_baseline
from baselines.graphrnn.run import run_graphrnn, load_model
from baselines.graphrnn.train import sample as sample_graphrnn
from config.config import Config, BaselineConfig, GraphRNNConfig

from dataset.manager import get_dataset_class
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
            self.name = f"{self.dataset}_{datetime.now().isoformat()}"
            self.root = RUNS_DIR / self.model_name / f"{self.dataset}" / f"{datetime.now().isoformat()}"
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
        test_data = dataset.get_data('test')
        trainer.fit(loader, test_data)
        config.save(self.root / "config")

    def resume(self):
        config = Config.from_file(self.root / "config" / f"config.yaml")
        dataset = self.dataset_class(config, self.root, name=self.dataset)
        trainer = Trainer.load(config, self.root, dataset.input_dim, dataset.output_dim)
        loader = dataset.get_loader('train')
        test_data = dataset.get_data('test')
        trainer.fit(loader, test_data)
        config.save(self.root / "config")

    def sample(self, num_samples):
        config = Config.from_file(self.root / "config" / f"config.yaml")
        dataset = self.dataset_class(config, self.root, name=self.dataset)
        trainer = Trainer.load(config, self.root, dataset.input_dim, dataset.output_dim, best=True)
        samples = trainer.sample(num_samples=num_samples)
        return samples


class BaselineExperiment(Experiment):
    @classmethod
    def load(cls, root):
        assert root is not None
        dataset = Path(root).parts[-2]
        model_name, metric = Path(root).parts[-3].split("_")
        return cls(model_name, metric, dataset, root=root)

    def __init__(self, model_name, metric, dataset, root=None):
        self.dataset = dataset
        self.model_name = model_name
        self.metric = metric
        self.dataset_class = get_dataset_class(dataset)

        if root is None:
            self.name = f"{self.dataset}_{datetime.now().isoformat()}"
            self.root = RUNS_DIR / f"{self.model_name}_{metric}" / f"{self.dataset}" / f"{datetime.now().isoformat()}"
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


class GraphRNNExperiment(Experiment):
    model_name = "GRAPHRNN"

    def __init__(self, dataset, root=None):
        self.dataset = dataset
        self.dataset_class = get_dataset_class(dataset)

        if root is None:
            self.name = f"{self.dataset}_{datetime.now().isoformat()}"
            self.root = RUNS_DIR / f"{self.model_name}" / f"{self.dataset}" / f"{datetime.now().isoformat()}"
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
        rnn_state_dict = torch.load(self.root / "ckpt" / f"rnn.pt", map_location=device)
        output_state_dict = torch.load(self.root / "ckpt" / f"output.pt", map_location=device)
        rnn, output = load_model(config, rnn_state_dict, output_state_dict)
        samples = sample_graphrnn(config, rnn, output, num_samples=num_samples)
        return samples
