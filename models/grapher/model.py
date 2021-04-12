import time

import torch
from torch import nn
from torch.nn import functional as F

from utils.constants import PAD, SOS, EOS


class RNN(nn.Module):
    def __init__(self, config, embed_dim, hidden_dim, num_tokens):
        super().__init__()

        self.config = config
        self.num_tokens = num_tokens
        self.rnn = nn.GRU(embed_dim, hidden_dim,
                          num_layers=config.num_layers,
                          dropout=config.dropout,
                          batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_tokens)
        self.num_tokens = num_tokens

    def forward(self, inputs, lengths, h0=None, log_softmax=True):
        batch_size, seq_len = inputs.size()

        x = nn.utils.rnn.pack_padded_sequence(inputs, lengths=lengths.cpu(), batch_first=True)
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
    def __init__(self, config, num_tokens, device):
        super().__init__()
        self.config = config
        self.num_tokens = num_tokens
        self.device = device

        self.embed = nn.Embedding(num_tokens, config.embed_dim, padding_idx=PAD)
        self.rnn1 = RNN(config, config.embed_dim, config.hidden_dim, num_tokens)
        self.rnn2 = RNN(config, config.embed_dim, config.hidden_dim, num_tokens)

    def forward_rnn(self, rnn, seq, lengths, h=None):
        targets = seq[:,1:]
        seq[:,lengths] = 0
        inputs = seq[:,:-1]
        outputs, h = rnn(inputs, lengths, h0=h)
        return outputs, targets, h

    def forward(self, batch):
        start = time.time()

        rnn1_seq, rnn2_seq, lengths = batch
        rnn1_seq.to(self.device)
        rnn2_seq.to(self.device)
        lengths.to(self.device)

        batch_size = rnn1_seq.size(0)

        rnn1_seq = self.embed(rnn1_seq)
        outputs1, targets1, h = self.forward_rnn(self.rnn1, rnn1_seq, lengths-1)

        rnn2_seq = self.embed(rnn2_seq)
        outputs2, targets2, _ = self.forward_rnn(self.rnn2, rnn2_seq, lengths-2, h=h)
        loss1 = F.cross_entropy(outputs1.view(-1, self.num_tokens), targets1.reshape(-1), ignore_index=PAD)
        loss2 = F.cross_entropy(outputs2.view(-1, self.num_tokens), targets2.reshape(-1), ignore_index=PAD)
        loss = loss1 + loss2

        return loss, {
            "loss": loss.item() / batch_size,
            "loss1": loss1.item() / batch_size,
            "loss2": loss2.item() / batch_size,
            "time": time.time() - start
        }

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
            first_endpoints, hs = self._sample_rnn1()
            last_endpoints = self._sample_rnn2(first_endpoints, hs[-1])
            edges = list(zip(first_endpoints, last_endpoints))

            samples.append(edges)
        return samples