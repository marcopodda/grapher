import torch
from torch import optim
from torch.optim import lr_scheduler


def get_device(config):
    use_cuda = config.device in ["cuda", "gpu"] and torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")


def get_scheduler(config, optimizer):
    scheduler_class = getattr(lr_scheduler, config.scheduler_class)
    return scheduler_class(optimizer, **config.scheduler_params)


def get_optimizer(config, model):
    optimizer_class = getattr(optim, config.optimizer_class)
    return optimizer_class(model.parameters(), **config.optimizer_params)


def is_duplicate(edges, graphlist):
    return edges in [list(G.edges()) for G in graphlist]