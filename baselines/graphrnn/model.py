import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


def binary_cross_entropy_weight(y_pred,
                                y,
                                has_weight=False,
                                weight_length=1,
                                weight_max=10):
    '''

    :param y_pred:
    :param y:
    :param weight_length: how long until the end of sequence
           shall we add weight
    :param weight_value: the magnitude that the weight is enhanced
    :return:
    '''
    if has_weight:
        weight = torch.ones(y.size(0), y.size(1), y.size(2))
        weight_linear = torch.arange(
            1, weight_length + 1) / weight_length * weight_max
        weight_linear = weight_linear.view(1, weight_length, 1).repeat(
            y.size(0), 1, y.size(2))
        weight[:, -1 * weight_length:, :] = weight_linear
        loss = F.binary_cross_entropy(y_pred, y, weight=weight)
    else:
        loss = F.binary_cross_entropy(y_pred, y)
    return loss


def sample_tensor(y, sample=True, thresh=0.5):
    # do sampling
    if sample:
        y_thresh = Variable(torch.rand(y.size()))
        y_result = torch.gt(y, y_thresh).float()
    # do max likelihood based on some threshold
    else:
        y_thresh = Variable(torch.ones(y.size()) * thresh)
        y_result = torch.gt(y, y_thresh).float()
    return y_result


def gumbel_softmax(logits, temperature, eps=1e-9):
    '''

    :param logits: shape: N*L
    :param temperature:
    :param eps:
    :return:
    '''
    # get gumbel noise
    noise = torch.rand(logits.size())
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    noise = Variable(noise)

    x = (logits + noise) / temperature
    x = F.softmax(x)
    return x


def gumbel_sigmoid(logits, temperature):
    '''

    :param logits:
    :param temperature:
    :param eps:
    :return:
    '''
    # get gumbel noise
    noise = torch.rand(logits.size())
    noise_logistic = torch.log(noise) - torch.log(1 - noise)
    noise = Variable(noise_logistic)

    x = (logits + noise) / temperature
    x = torch.sigmoid(x)
    return x


def sample_sigmoid(y, sample, device, thresh=0.5, sample_time=2):
    '''
        do sampling over unnormalized score
    :param y: input
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    # do sigmoid first
    y = torch.sigmoid(y)
    # do sampling
    if sample:
        if sample_time > 1:
            y_result = Variable(torch.rand(y.size(0), y.size(1),
                                           y.size(2))).to(device)
            # loop over all batches
            for i in range(y_result.size(0)):
                # do 'multi_sample' times sampling
                for j in range(sample_time):
                    y_thresh = Variable(torch.rand(y.size(1),
                                                   y.size(2))).to(device)
                    y_result[i] = torch.gt(y[i], y_thresh).float()
                    if (torch.sum(y_result[i]).data > 0).any():
                        break
                    # else:
                    #     print('all zero',j)
        else:
            y_thresh = Variable(torch.rand(y.size(0), y.size(1),
                                           y.size(2))).to(device)
            y_result = torch.gt(y, y_thresh).float()
    # do max likelihood based on some threshold
    else:
        y_thresh = Variable(
            torch.ones(y.size(0), y.size(1), y.size(2)) * thresh).to(device)
        y_result = torch.gt(y, y_thresh).float()
    return y_result


def sample_sigmoid_supervised(y_pred, y, current, y_len, sample_time=2):
    '''
        do sampling over unnormalized score
    :param y_pred: input
    :param y: supervision
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    # do sigmoid first
    y_pred = torch.sigmoid(y_pred)
    # do sampling
    y_result = Variable(
        torch.rand(y_pred.size(0), y_pred.size(1), y_pred.size(2)))
    # loop over all batches
    for i in range(y_result.size(0)):
        # using supervision
        if current < y_len[i]:
            while True:
                y_thresh = Variable(
                    torch.rand(y_pred.size(1), y_pred.size(2)))
                y_result[i] = torch.gt(y_pred[i], y_thresh).float()
                y_diff = y_result[i].data - y[i]
                if (y_diff >= 0).all():
                    break
        # supervision done
        else:
            # do 'multi_sample' times sampling
            for j in range(sample_time):
                y_thresh = Variable(
                    torch.rand(y_pred.size(1), y_pred.size(2)))
                y_result[i] = torch.gt(y_pred[i], y_thresh).float()
                if (torch.sum(y_result[i]).data > 0).any():
                    break
    return y_result


def sample_sigmoid_supervised_simple(y_pred, y, current, y_len, sample_time=2):
    '''
        do sampling over unnormalized score
    :param y_pred: input
    :param y: supervision
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    # do sigmoid first
    y_pred = torch.sigmoid(y_pred)
    # do sampling
    y_result = Variable(
        torch.rand(y_pred.size(0), y_pred.size(1), y_pred.size(2)))
    # loop over all batches
    for i in range(y_result.size(0)):
        # using supervision
        if current < y_len[i]:
            y_result[i] = y[i]
        # supervision done
        else:
            # do 'multi_sample' times sampling
            for j in range(sample_time):
                y_thresh = Variable(
                    torch.rand(y_pred.size(1), y_pred.size(2)))
                y_result[i] = torch.gt(y_pred[i], y_thresh).float()
                if (torch.sum(y_result[i]).data > 0).any():
                    break
    return y_result


# definition of terms
# h: hidden state of LSTM
# y: edge prediction, model output
# n: noise for generator
# l: whether an output is real or not, binary


# plain GRU model
class GRU_plain(nn.Module):
    def __init__(self,
                 input_size,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 device,
                 has_input=True,
                 has_output=False,
                 output_size=None):
        super(GRU_plain, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output
        self.device = device

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.GRU(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True)
        else:
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True)
        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size), nn.ReLU(),
                nn.Linear(embedding_size, output_size))

        self.relu = nn.ReLU()
        # initialize
        self.hidden = None  # need initialize before forward run

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform_(
                    param, gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        return Variable(
            torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(self.device)

    def forward(self, input_raw, pack=False, input_len=None):
        if self.has_input:
            input = self.input(input_raw)
            input = self.relu(input)
        else:
            input = input_raw
        if pack:
            input = pack_padded_sequence(input, lengths=input_len.cpu(), batch_first=True)
        output_raw, self.hidden = self.rnn(input, self.hidden)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        if self.has_output:
            output_raw = self.output(output_raw)
        # return hidden state at each time step
        return output_raw
