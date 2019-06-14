from .model import GRU_plain
from .train import train
from .args import Args
from .data import Graph_sequence_sampler_pytorch
import os
import torch
from utils.training import get_device


def run_graphrnn(config, dataset_name, exp_root, graphlist):
    config.update(max_num_node=graphlist.max_nodes)

    print('max number node: {}'.format(config.max_num_node))
    print('max previous node: {}'.format(config.max_prev_node))

    # dataset initialization
    dataset = Graph_sequence_sampler_pytorch(
        graphlist,
        max_prev_node=config.max_prev_node,
        max_num_node=config.max_num_node)
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
        [1.0 / len(dataset) for i in range(len(dataset))],
        num_samples=config.batch_size * config.batch_ratio,
        replacement=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=sample_strategy)

    config.update(max_prev_node=dataset.max_prev_node)

    device = get_device(config)

    # model initialization
    rnn = GRU_plain(
        input_size=config.max_prev_node,
        embedding_size=config.embedding_size_rnn,
        hidden_size=config.hidden_size_rnn,
        num_layers=config.num_layers,
        has_input=True,
        has_output=True,
        output_size=config.hidden_size_rnn_output).to(device)

    output = GRU_plain(
        input_size=1,
        embedding_size=config.embedding_size_rnn_output,
        hidden_size=config.hidden_size_rnn_output,
        num_layers=config.num_layers,
        has_input=True,
        has_output=True,
        output_size=1).to(device)

    samples = train(config, exp_root, dataloader, rnn, output, device)
    return samples
