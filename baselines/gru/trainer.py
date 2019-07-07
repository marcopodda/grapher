import numpy as np

import torch

from .model import Model, Loss
from utils.training import get_device, get_scheduler, get_optimizer


class GRUTrainer:
    @classmethod
    def load(cls, config, exp_root, input_dim, output_dim, best=False):
        filename = "best.pt" if best else "last.pt"
        path = exp_root / "ckpt" / filename
        device = get_device(config)
        ckpt = torch.load(path, map_location=device)

        trainer = cls(config, exp_root, input_dim, output_dim)
        trainer.model.load_state_dict(ckpt["model"])
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        trainer.scheduler.load_state_dict(ckpt["scheduler"])
        trainer.loss.load_state_dict(ckpt["loss"])
        trainer.best_loss = ckpt["best_loss"]
        trainer.losses = ckpt["losses"]
        trainer.current_epoch = ckpt['epoch'] + 1
        trainer.best_losses = ckpt['best_losses']
        return trainer

    def __init__(self, config, exp_root, input_dim, output_dim):
        self.exp_root = exp_root
        self.config = config
        self.model = Model(config, input_dim, output_dim)
        self.loss = Loss(output_dim)
        self.optimizer = get_optimizer(config, self.model)
        self.scheduler = get_scheduler(config, self.optimizer)
        self.device = get_device(config)

        self.losses = []
        self.current_epoch = 0
        self.best_loss = np.float('inf')
        self.best_losses = []

    def _train_epoch(self, loader):
        self.model.train()
        self.model.to(self.device)

        epoch_loss = 0

        for batch in loader:
            x, y, lengths = batch
            x = x.to(self.device)
            y = y.to(self.device)
            lengths = lengths.to(self.device)

            self.optimizer.zero_grad()
            outputs, h = self.model(x, lengths)
            loss = self.loss(outputs, y)
            epoch_loss += loss.item()

            loss.backward()
            self.optimizer.step()

        return epoch_loss / len(loader)

    def fit(self, loader):
        self.model.train()

        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            epoch_loss = self._train_epoch(loader)

            self.losses.append(epoch_loss)

            self.save(best=False)

            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.best_losses.append(epoch_loss)
                self.save(best=True)

            self.best_losses.append(self.best_loss)
            if self.best_loss < self.config.stop_loss:
                break

            self.log_epoch()

    def sample(self, num_samples):
        self.model.to('cpu')
        samples = self.model.sample(num_samples)
        self.model.to(self.device)
        return samples

    def save(self, best=False):
        filename = "best.pt" if best else "last.pt"
        path = self.exp_root / "ckpt" / filename
        torch.save({
            "epoch": self.current_epoch,
            "best_loss": self.best_loss,
            "best_losses": self.best_losses,
            "losses": self.losses,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "loss": self.loss.state_dict()
        }, path)

    def log_epoch(self):
        msg = f"{self.current_epoch:06d} - " + \
            f"loss: {self.losses[-1]:.6f} - " + \
            f"best loss: {self.best_loss:.6f}"

        print(msg)
