import torch


def get_device(config):
    use_cuda = config.device in ["cuda", "gpu"] and torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")