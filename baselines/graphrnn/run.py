from .model import GRU_plain
from .train import train
from .args import Args
from .data import Graph_sequence_sampler_pytorch
import os
import torch


def run_graphrnn(dataset_name, exp_root, graphlist):
    # All necessary arguments are defined in args.py
    args = Args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('CUDA', args.cuda)
    print('File name prefix', args.fname)

    args.graph_type = dataset_name

    args.fname = "GraphRNN_RNN_{}_".format(dataset_name)
    args.fname_pred = args.fname + '_pred_'
    args.fname_train = args.fname + '_train_'
    args.fname_test = args.fname + '_test_'

    # check if necessary directories exist
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.isdir(args.graph_save_path):
        os.makedirs(args.graph_save_path)
    if not os.path.isdir(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    if not os.path.isdir(args.timing_save_path):
        os.makedirs(args.timing_save_path)
    if not os.path.isdir(args.figure_prediction_save_path):
        os.makedirs(args.figure_prediction_save_path)

    args.max_num_node = graphlist.max_nodes

    print('max number node: {}'.format(args.max_num_node))
    print('max previous node: {}'.format(args.max_prev_node))

    # dataset initialization
    dataset = Graph_sequence_sampler_pytorch(
        graphlist,
        max_prev_node=args.max_prev_node,
        max_num_node=args.max_num_node)
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
        [1.0 / len(dataset) for i in range(len(dataset))],
        num_samples=args.batch_size * args.batch_ratio,
        replacement=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=sample_strategy)

    args.max_prev_node = dataset.max_prev_node

    # model initialization
    rnn = GRU_plain(
        input_size=args.max_prev_node,
        embedding_size=args.embedding_size_rnn,
        hidden_size=args.hidden_size_rnn,
        num_layers=args.num_layers,
        has_input=True,
        has_output=True,
        output_size=args.hidden_size_rnn_output)

    output = GRU_plain(
        input_size=1,
        embedding_size=args.embedding_size_rnn_output,
        hidden_size=args.hidden_size_rnn_output,
        num_layers=args.num_layers,
        has_input=True,
        has_output=True,
        output_size=1)

    samples = train(exp_root, args, dataloader, rnn, output)
    return samples
