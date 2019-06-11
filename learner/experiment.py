import os
from datetime import datetime
from pathlib import Path

import torch

from config.config import Config
from dataset.manager import TUData
from .trainer import Trainer
from .evaluator import Evaluator


RUNS_DIR = Path('RUNS')


class Experiment:
    @classmethod
    def load(cls, root):
        assert root is not None
        dataset = Path(root).parts[-2]
        return cls(dataset, root=root)

    def __init__(self, dataset, root=None):
        self.dataset = dataset
        if root is None:
            self.name = f"{self.dataset}_{datetime.now().isoformat()}"

            self.root = RUNS_DIR / f"{self.dataset}" / f"{datetime.now().isoformat()}"
            if not self.root.exists():
                os.makedirs(self.root)
        else:
            self.root = Path(root)
            self.name = "_".join(self.root.parts[-2:])

        if not self.root.exists():
            os.makedirs(self.root)

        if not (self.root / "ckpt").exists():
            os.makedirs(self.root / "ckpt")

        if not (self.root / "data").exists():
            os.makedirs(self.root / "data")

        if not (self.root / "config").exists():
            os.makedirs((self.root / "config"))

        if not (self.root / "samples").exists():
            os.makedirs((self.root / "samples"))

        if not (self.root / "evaluation").exists():
            os.makedirs((self.root / "evaluation"))

    def train(self):
        config = Config.from_file(f"config_{self.dataset}.yaml")
        config.save(self.root / "config")
        dataset = TUData(config, self.root, name=self.dataset)
        trainer = Trainer(config, self.root, dataset.input_dim, dataset.output_dim)
        loader = dataset.get_loader('train')
        trainer.fit(loader)
        trainer.sample(config.num_samples, final=True)

    def resume(self):
        config = Config.from_file(self.root / "config" / f"config.yaml")
        dataset = TUData(config, self.root, name=self.dataset)
        trainer = Trainer.load(config, self.root, dataset.input_dim, dataset.output_dim)
        loader = dataset.get_loader('train')
        trainer.fit(loader)
        trainer.sample(config.num_samples, final=True)

    def evaluate(self):
        config = Config.from_file(self.root / "config" / f"config.yaml")
        dataset = TUData(config, self.root, name=self.dataset)
        evaluator = Evaluator(config, self.root)
        test_data = dataset.get_data('test')
        evaluator.evaluate(test_data)
