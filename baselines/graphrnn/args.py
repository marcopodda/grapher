# program configuration


class Args():
    def __init__(self):

        # if none, then auto calculate
        self.max_num_node = None  # max number of nodes in a graph
        self.max_prev_node = None  # max previous node that looks back

        # hidden size for main RNN
        self.hidden_size_rnn = 64
        self.hidden_size_rnn_output = 16  # hidden size for output RNN
        self.embedding_size_rnn = 32
        self.embedding_size_rnn_output = 8  # the embedding size for output rnn

        # the embedding size for output (VAE/MLP)
        self.embedding_size_output = 32

        # normal: 32, and the rest should be changed accordingly
        self.batch_size = 32

        self.test_batch_size = 32
        self.test_total_size = 1000
        self.num_layers = 3

        # training config
        self.num_workers = 4  # num workers to load data, default 4

        # how many batches of samples per epoch, default 32, e.g., 1 epoch = 32
        # batches
        self.batch_ratio = 32
        self.epochs = 1000  # now one epoch means self.batch_ratio x batch_size
        self.epochs_test_start = 1000
        self.epochs_test = 1000
        self.epochs_log = 1000
        self.epochs_save = 1000

        self.lr = 0.003
        self.milestones = [400, 1000]
        self.lr_rate = 0.3

        self.sample_time = 2  # sample time in each time step, when validating
