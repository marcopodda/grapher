import torch
import numpy as np
import networkx as nx
from torch import nn
from torch.utils.data import Dataset
from utils.data import to_sorted_tensor, reverse_argsort, pad_left, pad_right
from utils.constants import SOS, EOS


class GRUDataset(Dataset):
    def __init__(self, config, graphlist):
        super().__init__()
        self.config = config
        self.graphlist = list(graphlist._graphs)

    def __len__(self):
        return len(self.graphlist)

    def __getitem__(self, index):
        G = self.graphlist[index]
        adj = nx.to_numpy_array(G)
        adj_vector = adj[np.tril_indices(adj.shape[0], k=-1)]
        return (adj_vector + 3).flatten().tolist()

    @property
    def input_dim(self):
        return 5

    @property
    def output_dim(self):
        return 5


class GRUDataCollator:
    def __init__(self, config):
        self.config = config

    def __call__(self, seqs):
        # pad with SOS and EOS
        inputs = [pad_left(x, SOS) for x in seqs]
        outputs = [pad_right(x, EOS) for x in seqs]

        # calculate descending order
        lengths = [len(x) for x in inputs]
        order = reverse_argsort(lengths)
        # sort in descending order
        inputs = to_sorted_tensor(inputs, order)
        outputs = to_sorted_tensor(outputs, order)
        lengths = torch.LongTensor([lengths[i] for i in order])

        # 0-pad sequences
        input_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        output_padded = nn.utils.rnn.pad_sequence(outputs, batch_first=True)

        return input_padded, output_padded, lengths
