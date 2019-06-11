import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .graph import encode_graph


class GraphDataset(Dataset):
    def __init__(self, config, graphlist):
        super().__init__()
        self.config = config
        self.graphlist = graphlist

    def __len__(self):
        return len(self.graphlist)

    def __getitem__(self, index):
        G = self.graphlist[index]
        l1, l2 = encode_graph(G, self.config.bfs_order)
        return l1, l2

    @property
    def input_dim(self):
        # +3 because we are also counting pad, sos and eos tokens
        return self.graphlist.max_nodes + 3

    @property
    def output_dim(self):
        # +3 because we are also counting pad, sos and eos tokens
        return self.graphlist.max_nodes + 3


class GraphDataCollator:
    PAD = 0
    SOS = 1
    EOS = 2

    def __init__(self, config):
        self.config = config

    def __call__(self, batch):
        s1, s2 = zip(*batch)

        inputs = [(self.SOS,) + x for x in s1]
        shifted_inputs = [x + (self.EOS,) for x in s1]
        outputs = [x + (self.EOS,) for x in s2]
        lengths = [len(x) for x in inputs]

        # sort in descending order
        order = self.argsort(lengths)
        inputs = [torch.LongTensor(inputs[i]) for i in order]
        shifted_inputs = [torch.LongTensor(shifted_inputs[i]) for i in order]
        outputs = [torch.LongTensor(outputs[i]) for i in order]
        lengths = [lengths[i] for i in order]

		# pad sequences
        input_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        shifted_input_padded = nn.utils.rnn.pad_sequence(shifted_inputs, batch_first=True)
        output_padded = nn.utils.rnn.pad_sequence(outputs, batch_first=True)

        return input_padded, shifted_input_padded, output_padded, torch.LongTensor(lengths)

    def argsort(self, lengths):
        lengths = np.array(lengths)
        return (-lengths).argsort()