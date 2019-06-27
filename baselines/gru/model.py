import networkx as nx

import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, config, input_dim, output_dim):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(input_dim, config.embed_dim)
        self.gru = nn.GRU(config.embed_dim, config.hidden_dim)
        self.linear = nn.Linear(config.hidden_dim, output_dim)

    def forward(self, x, lengths, h0=None):
        x = self.embed(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        x, h = self.gru(x, h0)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.linear(x)

        return x, h

    def _sample(self):
        model = self

        step = 0
        max_length = self.config.max_num_edges
        temperature = self.config.temperature

        sample = []
        hs = []

        with torch.no_grad():
            h = None
            inputs = torch.LongTensor([1])  # SOS
            lengths = torch.LongTensor([1])
            while step < max_length:
                if inputs.dim() == 1:
                    inputs = inputs.unsqueeze(0)

                outputs, h = model(inputs, lengths, h)
                probs = F.softmax(outputs.squeeze(0) / temperature, dim=1)
                inputs = torch.multinomial(probs, 1).reshape(1, -1)

                if inputs.item() == 2:
                    break

                sample.append(inputs.item())
                hs.append(h)
                step += 1

        return sample, hs

    def sample(self, train_data, i2e, num_samples=1000):
        samples = []

        while len(samples) < num_samples:
            seq, hs = self._sample()
            edges = [i2e[i] for i in seq]

            if edges in samples:
                # remove duplicates
                continue

            if edges in train_data:
                # remove duplicates in training data
                continue

            samples.append(nx.Graph(edges))

        return samples[:num_samples]


class Loss(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, targets):
        return self.ce(x.view(-1, self.output_dim), targets.view(-1))
