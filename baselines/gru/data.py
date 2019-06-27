import torch
from torch import nn
from torch.utils.data import Dataset
from utils.data import SOS, EOS, to_sorted_tensor, reverse_argsort, pad_left, pad_right


class GRUDataset(Dataset):
    def __init__(self, config, e2i, graphlist):
        super().__init__()
        self.config = config
        self.e2i = e2i
        self.graphlist = graphlist

    def __len__(self):
        return len(self.graphlist)

    def __getitem__(self, index):
        edges = list(self.graphlist[index].edges())
        seq = [self.e2i[e] for e in edges]
        return seq

    @property
    def input_dim(self):
        # +3 because we are also counting pad, sos and eos tokens
        return self.graphlist.max_nodes + 3

    @property
    def output_dim(self):
        # +3 because we are also counting pad, sos and eos tokens
        return self.graphlist.max_nodes + 3


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
