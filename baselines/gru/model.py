import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils.constants import EOS, SOS, PAD


def decode_adj(output):
    tril = []
    for i in range(1, len(output)+1):
        t = output[:i]
        if len(tril) > 0 and len(t) < len(tril[-1]):
            break
        tril.append(t.tolist())
        output = output[len(t):]
    n = len(tril[-1]) + 1
    A = np.zeros((n, n))
    A[np.tril_indices(n, k=-1)] = [x for sub in tril for x in sub]
    return A + A.T
    # max_prev_node = adj_output.shape[1]
    # adj = np.zeros((adj_output.shape[1], adj_output.shape[1]))
    # for i in range(adj_output.shape[1]):
    #     input_start = max(0, i - max_prev_node + 1)
    #     input_end = i + 1
    #     output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
    #     output_end = max_prev_node
    #     adj[i, input_start:input_end] = adj_output[i, ::-1][
    #         output_start:output_end]  # reverse order
    # adj_full = np.zeros((adj_output.shape[1] + 1, adj_output.shape[1] + 1))
    # n = adj_full.shape[0]
    # adj_full[1:n, 0:n - 1] = np.tril(adj, 0)
    # adj_full = adj_full + adj_full.T

    # return adj_full


class Model(nn.Module):
    def __init__(self, config, input_dim, output_dim):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(input_dim, config.embed_dim)
        self.gru = nn.GRU(config.embed_dim, config.hidden_dim,
                          num_layers=config.num_layers,
                          dropout=config.dropout)
        self.linear = nn.Linear(config.hidden_dim, output_dim)

    def forward(self, x, lengths, h0=None):
        x = self.embed(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths=lengths.cpu(), batch_first=True)
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
            inputs = torch.LongTensor([SOS])
            lengths = torch.LongTensor([1])
            while step < max_length:
                if inputs.dim() == 1:
                    inputs = inputs.unsqueeze(0)

                outputs, h = model(inputs, lengths, h)
                probs = F.softmax(outputs.squeeze(0) / temperature, dim=1)
                inputs = torch.multinomial(probs, 1).reshape(1, -1)

                if inputs.item() == EOS:
                    break

                sample.append(inputs.item())
                hs.append(h)
                step += 1

        return np.array(sample), hs

    def sample(self, num_samples=1000):
        samples = []

        while len(samples) < num_samples:
            seq, hs = self._sample()
            try:
                adj = decode_adj(seq - 3)
                G = nx.from_numpy_array(adj)
                if G.number_of_nodes() > 0:
                    samples.append(list(G.edges()))
            except:
                continue

        return samples


class Loss(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        return self.ce(outputs.view(-1, self.output_dim), targets.view(-1))
