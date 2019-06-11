import numpy as np

import torch
from torch import optim
from torch.optim import lr_scheduler
from .model import Model, Loss
from dataset.graph import decode_graphs


def get_scheduler(config, optimizer):
    scheduler_class = getattr(lr_scheduler, config.scheduler_class)
    return scheduler_class(optimizer, **config.scheduler_params)


def get_optimizer(config, model):
    optimizer_class = getattr(optim, config.optimizer_class)
    return optimizer_class(model.parameters(), **config.optimizer_params)


class Trainer:
    @classmethod
    def load(cls, config, exp_root, input_dim, output_dim, best=False):
        filename = "best.pt" if best else "last.pt"
        path = exp_root / "ckpt" / filename
        ckpt = torch.load(path)

        trainer = cls(config, exp_root, input_dim, output_dim)
        trainer.model.load_state_dict(ckpt["model"])
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        trainer.scheduler.load_state_dict(ckpt["scheduler"])
        trainer.loss1.load_state_dict(ckpt["loss1"])
        trainer.loss2.load_state_dict(ckpt["loss2"])
        trainer.best_loss = ckpt["best_loss"]
        trainer.losses = ckpt["losses"]
        trainer.current_epoch = ckpt['epoch'] + 1
        return trainer

    def __init__(self, config, exp_root, input_dim, output_dim):
        self.exp_root = exp_root
        self.config = config
        self.model = Model(config, input_dim, output_dim)
        self.loss1 = Loss(output_dim)
        self.loss2 = Loss(output_dim)
        self.optimizer = get_optimizer(config, self.model)
        self.scheduler = get_scheduler(config, self.optimizer)

        self.losses = []
        self.current_epoch = 0
        self.best_loss = np.float('inf')

    def _run_epoch(self, loader):
        epoch_loss1 = 0
        epoch_loss2 = 0

        for batch in loader:
            i1, i2, i3, lengths = batch
            self.optimizer.zero_grad()
            out1, out2 = self.model(i1, i2, lengths)

            l1 = self.loss1(out1, i2)
            l2 = self.loss2(out2, i3)
            loss = l1 + l2

            epoch_loss1 += l1.item()
            epoch_loss2 += l2.item()

            loss.backward()
            self.optimizer.step()

        return epoch_loss1, epoch_loss2

    def fit(self, loader):
        self.model.train()

        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            epoch_loss1, epoch_loss2 = self._run_epoch(loader)
            total_loss = epoch_loss1 + epoch_loss2

            self.losses.append(total_loss)
            self.save(best=False)

            if total_loss < self.best_loss:
                self.best_loss = total_loss
                self.save(best=True)

            if epoch >= 10 and epoch % self.config.sample_interval == 0:
                self.sample(self.config.num_intermediate_samples)

            print(f"{epoch:06d}: {epoch_loss1 / len(loader):.6f} - {epoch_loss2 / len(loader):.6f}")

    def sample(self, num_samples, final=False):
        self.model.eval()
        samples = self.model.sample(num_samples)
        Gs = decode_graphs(samples)
        filename = f"final_samples.pt" if final else f"{self.current_epoch:06d}_samples.pt"
        torch.save(Gs, self.exp_root / "samples" / filename)

    def save(self, best=False):
        filename = "best.pt" if best else "last.pt"
        path = self.exp_root / "ckpt" / filename
        torch.save({
            "epoch": self.current_epoch,
            "best_loss": self.best_loss,
            "losses": self.losses,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "loss1": self.loss1.state_dict(),
            "loss2": self.loss2.state_dict(),
        }, path)
