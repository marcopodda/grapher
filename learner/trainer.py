import numpy as np

import torch
from .model import Model, Loss
from utils.training import get_device, get_scheduler, get_optimizer


class Trainer:
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
        trainer.loss1.load_state_dict(ckpt["loss1"])
        trainer.loss2.load_state_dict(ckpt["loss2"])
        trainer.best_loss = ckpt["best_loss"]
        trainer.losses1 = ckpt["losses1"]
        trainer.losses2 = ckpt["losses2"]
        trainer.current_epoch = ckpt['epoch'] + 1
        trainer.best_losses = ckpt['best_losses']
        return trainer

    def __init__(self, config, exp_root, input_dim, output_dim):
        self.exp_root = exp_root
        self.config = config
        self.model = Model(config, input_dim, output_dim)
        self.loss1 = Loss(output_dim)
        self.loss2 = Loss(output_dim)
        self.optimizer = get_optimizer(config, self.model)
        self.scheduler = get_scheduler(config, self.optimizer)
        self.device = get_device(config)

        self.losses1 = []
        self.losses2 = []

        self.current_epoch = 0
        self.best_loss = np.float('inf')
        self.best_losses = []

    def _train_epoch(self, loader):
        self.model.train()
        self.model.to(self.device)

        epoch_loss1 = 0
        epoch_loss2 = 0

        for batch in loader:
            i1, i2, i3, lengths = batch

            i1 = i1.to(self.device)
            i2 = i2.to(self.device)
            i3 = i3.to(self.device)
            lengths = lengths.to(self.device)

            self.optimizer.zero_grad()
            out1, out2 = self.model(i1, i2, lengths)

            l1 = self.loss1(out1, i2)
            l2 = self.loss2(out2, i3)
            loss = l1 + l2

            epoch_loss1 += l1.item()
            epoch_loss2 += l2.item()

            loss.backward()
            self.optimizer.step()

        return epoch_loss1 / len(loader), epoch_loss2 / len(loader)

    def fit(self, loader, order=None):
        self.model.train()

        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            epoch_loss1, epoch_loss2 = self._train_epoch(loader)
            total_loss = epoch_loss1 + epoch_loss2

            self.losses1.append(epoch_loss1)
            self.losses2.append(epoch_loss2)

            self.save(best=False)

            if total_loss < self.best_loss:
                self.best_loss = total_loss
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
            "losses1": self.losses1,
            "losses2": self.losses2,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "loss1": self.loss1.state_dict(),
            "loss2": self.loss2.state_dict(),
            "best_losses": self.best_losses
        }, path)

    def log_epoch(self):
        msg = f"{self.current_epoch:06d} - " + \
            f"loss1: {self.losses1[-1]:.6f} - " + \
            f"loss2: {self.losses2[-1]:.6f} - " + \
            f"total: {self.losses1[-1] + self.losses2[-1]:.6f} - " + \
            f"best_loss: {self.best_loss:.6f}"

        print(msg)