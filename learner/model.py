import torch
from torch import nn
from torch.nn import functional as F

import networkx as nx

from utils.constants import PAD, SOS, EOS
from utils.graphs import max_connected_comp

class RNN(nn.Module):
    def __init__(self, config, input_dim, embed_dim, hidden_dim, output_dim):
        super().__init__()

        self.config = config
        self.output_dim = output_dim
        self.embed = nn.Embedding(input_dim, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim,
                          num_layers=config.num_layers,
                          dropout=config.dropout,
                          batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, inputs, lengths, h0=None, log_softmax=True):
        batch_size, seq_len = inputs.size()

        x = self.embed(inputs)

        x = nn.utils.rnn.pack_padded_sequence(x, lengths=lengths.cpu(), batch_first=True)
        outputs, h = self.rnn(x, h0)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        outputs = outputs.contiguous()
        outputs = outputs.view(-1, outputs.size(2))

        outputs = self.linear(outputs)

        if log_softmax:
            outputs = F.log_softmax(outputs, dim=1)
            outputs = outputs.view(batch_size, seq_len, -1)

        return outputs, h


class Model(nn.Module):
    def __init__(self, config, input_dim, output_dim):
        super().__init__()
        self.config = config
        self.output_dim = output_dim

        self.rnn1 = RNN(config, input_dim, config.embed_dim, config.hidden_dim, output_dim)
        self.rnn2 = RNN(config, input_dim, config.embed_dim, config.hidden_dim, output_dim)

    def forward(self, input1, input2, lengths):
        outputs1, h = self.rnn1(input1, lengths)
        outputs2, _ = self.rnn2(input2, lengths, h0=h)

        return outputs1, outputs2

    def _sample_rnn1(self):
        model = self.rnn1

        step = 0
        max_length = self.config.max_num_edges
        temperature = self.config.temperature1

        sample = []
        hs = []

        with torch.no_grad():
            h = None
            inputs = torch.LongTensor([SOS])
            lengths = torch.LongTensor([1])
            while step < max_length:
                if inputs.dim() == 1:
                    inputs = inputs.unsqueeze(0)

                outputs, h = model(inputs, lengths, h, log_softmax=False)
                probs = F.softmax(outputs / temperature, dim=1)
                inputs = torch.multinomial(probs, 1).reshape(1, -1)

                if inputs.item() == EOS:
                    break

                sample.append(inputs.item())
                hs.append(h)
                step += 1

        return sample, hs

    def _sample_rnn2(self, inputs, h):
        temperature = self.config.temperature2

        with torch.no_grad():
            model = self.rnn2

            lengths = torch.LongTensor([len(inputs)])
            inputs = torch.LongTensor(inputs).unsqueeze(0)
            outputs, h = model(inputs, lengths, h, log_softmax=False)
            probs = F.softmax(outputs / temperature, dim=1)
            outputs = torch.multinomial(probs, 1)

            outputs = outputs.numpy().tolist()
            if isinstance(outputs[0], list):
                outputs = [o[0] for o in outputs]

            return outputs

    def sample(self, num_samples=1000):
        samples = []

        while len(samples) < num_samples:
            inputs, hs = self._sample_rnn1()
            outputs = self._sample_rnn2(inputs, hs[-1])
            edges = list(zip(inputs, outputs))
            G = nx.Graph(edges)
            G = max_connected_comp(G)
            samples.append(G)
        return samples


class Loss(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, outputs, targets):
        # TRICK 3 ********************************
        # before we calculate the negative log likelihood, we need to mask out the activations
        # this means we don't want to take into account padded items in the output vector
        # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        # and calculate the loss on that.

        # flatten all the labels
        targets = targets.view(-1)

        # flatten all predictions
        outputs = outputs.view(-1, self.output_dim)

        # create a mask by filtering out all tokens that ARE NOT the padding token
        mask = (targets > PAD).float()

        # count how many tokens we have
        nb_tokens = int(torch.sum(mask).item())

        # pick the values for the label and zero out the rest with the mask
        outputs = outputs[range(outputs.size(0)), targets] * mask

        # compute cross entropy loss which ignores all <PAD> tokens
        ce_loss = -torch.sum(outputs) / nb_tokens

        return ce_loss
