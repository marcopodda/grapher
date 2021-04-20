import networkx as nx
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from .model import binary_cross_entropy_weight, sample_sigmoid
from .data import decode_adj
from utils.training import get_device, is_duplicate
from dataset.graph import GraphList


def get_graph(adj):
    '''
    get a graph from zero-padded adj
    :param adj:
    :return:
    '''
    # remove all zeros rows and columns
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]
    adj = np.asmatrix(adj)
    G = nx.from_numpy_matrix(adj)
    return G


def train_rnn_epoch(epoch, config, rnn, output, data_loader, optimizer_rnn,
                    optimizer_output, scheduler_rnn, scheduler_output, device):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float().to(device)
        y_unsorted = data['y'].float().to(device)
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :].to(device)
        y_unsorted = y_unsorted[:, 0:y_len_max, :].to(device)
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        # output.hidden = output.init_hidden(
        #     batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # sort input
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        sort_index = sort_index.to(device)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index).to(device)
        y = torch.index_select(y_unsorted, 0, sort_index).to(device)

        # input, output for output rnn module
        # a smart use of pytorch builtin function:
        # pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y, lengths=y_len, batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0) - 1, -1, -1)]
        idx = torch.LongTensor(idx).to(device)
        y_reshape = y_reshape.index_select(0, idx).to(device)
        y_reshape = y_reshape.view(y_reshape.size(0), y_reshape.size(1), 1)

        output_x = torch.cat(
            (torch.ones(y_reshape.size(0), 1, 1).to(device), y_reshape[:, 0:-1, 0:1]),
            dim=1).to(device)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin) - 1, 0, -1):
            count_temp = np.sum(
                output_y_len_bin[i:])  # count how many y_len is above i
            output_y_len.extend([min(i, y.size(2))] * count_temp)
        # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x = Variable(x).to(device)
        y = Variable(y).to(device)
        output_x = Variable(output_x).to(device)
        output_y = Variable(output_y).to(device)

        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        h = pack_padded_sequence(h, lengths=y_len, batch_first=True).data  # get packed hidden vector
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).to(device)
        h = h.index_select(0, idx).to(device)
        hidden_null = Variable(
            torch.zeros(config.num_layers - 1, h.size(0), h.size(1))).to(device)
        output.hidden = torch.cat(
            (h.view(1, h.size(0), h.size(1)), hidden_null),
            dim=0)  # num_layers, batch_size, hidden_size
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = torch.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, lengths=output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y, lengths=output_y_len, batch_first=True)
        output_y = pad_packed_sequence(output_y, batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, output_y)
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()

        feature_dim = y.size(1) * y.size(2)
        loss_sum += loss.item() * feature_dim
    return loss_sum / (batch_idx + 1)


def test_rnn_epoch(config, rnn, output, device, test_batch_size=16):
    print("-------------", device)
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(config.max_num_node)
    y_pred_long = Variable(
        torch.zeros(test_batch_size, max_num_node,
                    config.max_prev_node)).to(device)  # discrete prediction
    x_step = Variable(torch.ones(test_batch_size, 1,
                                 config.max_prev_node)).to(device)
    for i in range(max_num_node):
        h = rnn(x_step)
        # output.hidden = h.permute(1,0,2)
        hidden_null = Variable(
            torch.zeros(config.num_layers - 1, h.size(0), h.size(2))).to(device)
        output.hidden = torch.cat((h.permute(1, 0, 2), hidden_null),
                                  dim=0)  # num_layers, batch_size, hidden_size
        x_step = Variable(torch.zeros(test_batch_size, 1,
                                      config.max_prev_node)).to(device)
        output_x_step = Variable(torch.ones(test_batch_size, 1, 1)).to(device)
        for j in range(min(config.max_prev_node, i + 1)):
            output_y_pred_step = output(output_x_step)
            output_x_step = sample_sigmoid(
                output_y_pred_step, sample=True, device=device, sample_time=1)
            x_step[:, :, j:j + 1] = output_x_step
            output.hidden = Variable(output.hidden.data).to(device)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).to(device)
    y_pred_long_data = y_pred_long.data.long()

    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred)  # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return list(G_pred_list[0].edges())


# train function for RNN
def train(config, exp_root, dataloader, rnn, output):
    device = get_device(config)
    # check if load existing model
    epoch = 1

    # initialize optimizer
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=config.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=config.lr)

    scheduler_rnn = MultiStepLR(
        optimizer_rnn, milestones=config.milestones, gamma=config.lr_rate)
    scheduler_output = MultiStepLR(
        optimizer_output, milestones=config.milestones, gamma=config.lr_rate)

    # start main loop
    while epoch <= config.epochs:

        loss = train_rnn_epoch(
            epoch, config, rnn, output, dataloader, optimizer_rnn,
            optimizer_output, scheduler_rnn, scheduler_output, device)

        print('Epoch: {}/{}, train loss: {:.6f}'.format(
            epoch, config.epochs, loss / len(dataloader)))

        epoch += 1

    fname = exp_root / "ckpt" / "rnn.pt"
    torch.save(rnn.state_dict(), fname)
    fname = exp_root / "ckpt" / "output.pt"
    torch.save(output.state_dict(), fname)


def sample(config, rnn, output, num_samples):
    samples = []
    device = get_device(config)

    while len(samples) < num_samples:

        edges = test_rnn_epoch(
            config,
            rnn,
            output,
            device,
            test_batch_size=256)

        samples.append(edges)

    return samples
