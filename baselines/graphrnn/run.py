from .model import GRU_plain
from .train import train
from .data import Graph_sequence_sampler_pytorch
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
    # model initialization
    rnn, output = load_model(config)

    train(config, exp_root, dataloader, rnn, output)


def load_model(config, rnn_state_dict=None, output_state_dict=None):
    device = get_device(config)

    # model initialization
    rnn = GRU_plain(
        input_size=config.max_prev_node,
        embedding_size=config.embedding_size_rnn,
        hidden_size=config.hidden_size_rnn,
        num_layers=config.num_layers,
        device=device,
        has_input=True,
        has_output=True,
        output_size=config.hidden_size_rnn_output).to(device)

    output = GRU_plain(
        input_size=1,
        embedding_size=config.embedding_size_rnn_output,
        hidden_size=config.hidden_size_rnn_output,
        num_layers=config.num_layers,
        device=device,
        has_input=True,
        has_output=True,
        output_size=1).to(device)

    if rnn_state_dict is not None:
        rnn.load_state_dict(rnn_state_dict)

    if output_state_dict is not None:
        output.load_state_dict(output_state_dict)

    return rnn, output
