from torch import nn


class GRUNet(nn.Module):
    def __init__(self, config, input_dim, output_dim):
        super().__init__()
        self.gru = nn.GRU()
        self.linear = nn.Linear()

    def forward(self, x):
        x = self.gru(x)
        x = self.linear(x)
        return x


class Loss(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, targets):
        return self.ce(x, targets)
