import os
from datetime import datetime
from pathlib import Path

import torch

from config.config import Config
from dataset.manager import get_dataset_class
from .trainer import Trainer
from .evaluator import Evaluator


RUNS_DIR = Path('RUNS')


def maybe_makedir(path):
    if not path.exists():
        os.makedirs(path)


class Experiment:
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
            self.root = RUNS_DIR / f"{self.dataset}" / f"{datetime.now().isoformat()}"
        else:
            self.root = Path(root)
            self.name = "_".join(self.root.parts[-2:])

        maybe_makedir(self.root)
        maybe_makedir(self.root / "ckpt")
        maybe_makedir(self.root / "data")
        maybe_makedir(self.root / "config")
        maybe_makedir(self.root / "samples")
        maybe_makedir(self.root / "evaluation")

    def train(self):
        config = Config.from_file(f"config_{self.dataset}.yaml")
        config.save(self.root / "config")
        dataset = self.dataset_class(config, self.root, name=self.dataset)
        trainer = Trainer(config, self.root, dataset.input_dim, dataset.output_dim)
        loader = dataset.get_loader('train')
        trainer.fit(loader)
        trainer.sample(config.num_samples, final=True)

    def resume(self):
        config = Config.from_file(self.root / "config" / f"config.yaml")
        dataset = self.dataset_class(config, self.root, name=self.dataset)
        trainer = Trainer.load(config, self.root, dataset.input_dim, dataset.output_dim)
        loader = dataset.get_loader('train')
        trainer.fit(loader)
        trainer.sample(config.num_samples, final=True)

    def evaluate(self):
        config = Config.from_file(self.root / "config" / f"config.yaml")
        dataset = self.dataset_class(config, self.root, name=self.dataset)
        evaluator = Evaluator(config, self.root)
        test_data = dataset.get_data('test')
        evaluator.evaluate(test_data)
