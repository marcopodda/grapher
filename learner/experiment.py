import os
from datetime import datetime
from pathlib import Path
from config.config import Config
from dataset.manager import TUData
from .trainer import Trainer


RUNS_DIR = Path('RUNS')


class Experiment:
    @classmethod
    def load(cls, dataset, root):
        assert root is not None
        return cls(dataset, root=root)

    def __init__(self, dataset, root=None):
        self.dataset = dataset
        if root is None:
            self.name = f"{self.dataset}_{datetime.now().isoformat()}"

            self.root = RUNS_DIR / self.name
            if not self.root.exists():
                os.makedirs(self.root)
        else:
            self.root = Path(root)
            self.name = self.root.parts[-1]

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

    def run(self):
        config = Config.from_file(f"config_{self.dataset}.yaml")
        config.save(self.root / "config")
        dataset = TUData(config, self.root, name=self.dataset)
        print(len(dataset))
        trainer = Trainer(config, self.root, dataset.input_dim, dataset.output_dim)
        loader = dataset.get_loader('train')
        trainer.fit(loader)
        trainer.sample(config.num_samples)

    def resume(self):
        config = Config.from_file(self.root / "config" / f"config.yaml")
        dataset = TUData(config, self.root, name=self.dataset)
        trainer, epoch = Trainer.load(config, self.root, dataset.input_dim, dataset.output_dim)
        loader = dataset.get_loader('train')
        trainer.fit(loader, start_epoch=epoch)
        trainer.sample(config.num_samples)
