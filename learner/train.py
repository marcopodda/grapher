import numpy as np
from torch import optim
from torch.optim import lr_scheduler
from .model import Model, Loss
from dataset.manager import TUData
from config.config import Config

def get_scheduler(config, optimizer):
    scheduler_class = getattr(lr_scheduler, config.scheduler_class)
    return scheduler_class(optimizer, **config.scheduler_params)


def get_optimizer(config, model):
    optimizer_class = getattr(optim, config.optimizer_class)
    return optimizer_class(model.parameters(), **config.optimizer_params)


class Trainer:
    def __init__(self, config, input_dim, output_dim):
        self.config = config
        self.model = Model(config, input_dim, output_dim)
        self.loss1 = Loss(output_dim)
        self.loss2 = Loss(output_dim)
        self.optimizer = get_optimizer(config, self.model)
        self.scheduler = get_scheduler(config, self.optimizer)

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

    def fit(self, loader, start_epoch=0):
        self.model.train()

        for epoch in range(start_epoch, self.config.max_epochs):
            epoch_loss1, epoch_loss2 = self._run_epoch(loader)
            total_loss = epoch_loss1 + epoch_loss2
            print(f"{epoch:06d}: {epoch_loss1 / len(loader):.6f} - {epoch_loss2 / len(loader):.6f}")

            if total_loss < self.best_loss:
                self.best_loss = total_loss
