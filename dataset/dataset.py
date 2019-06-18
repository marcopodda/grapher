import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .graph import encode_graph

from learner.skipgram import train_embeddings


PAD = 0
SOS = 1
EOS = 2


def to_sorted_tensor(lst, order):
    return [torch.LongTensor(lst[i]) for i in order]


def reverse_argsort(lst):
    arr = np.array(lst)
    return (-arr).argsort()


def pad_right(arr, pad):
    return arr + (pad,)


def pad_left(arr, pad):
    return (pad,) + arr


class GraphDataset(Dataset):
    def __init__(self, name, exp_root, config, graphlist):
        super().__init__()
        self.config = config
        self.graphlist = graphlist

        emb_data = []
        for G in graphlist:
            l1, _ = encode_graph(G, self.config.bfs_order)
            emb_data.append([str(i) for i in l1])
        
        train_embeddings(name, exp_root, config, emb_data)
        
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
    def __init__(self, config):
        self.config = config

    def __call__(self, batch):
        s1, s2 = zip(*batch)

        # pad with SOS and EOS
        inputs = [pad_left(x, SOS) for x in s1]
        shifted_inputs = [pad_right(x, EOS) for x in s1]
        outputs = [pad_right(x, EOS) for x in s2]

        # calculate descending order
        lengths = [len(x) for x in inputs]
        order = reverse_argsort(lengths)

        # sort in descending order
        inputs = to_sorted_tensor(inputs, order)
        shifted_inputs = to_sorted_tensor(shifted_inputs, order)
        outputs = to_sorted_tensor(outputs, order)
        lengths = torch.LongTensor([lengths[i] for i in order])

        # 0-pad sequences
        input_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        shifted_input_padded = nn.utils.rnn.pad_sequence(shifted_inputs, batch_first=True)
        output_padded = nn.utils.rnn.pad_sequence(outputs, batch_first=True)

        return input_padded, shifted_input_padded, output_padded, lengths


