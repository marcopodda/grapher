from .base import BaseConfigWithSerializer


class Config(BaseConfigWithSerializer):
    def set_defaults(self):
        Config._params.update(
            bfs_order=True,
            batch_size=16,
            max_num_nodes=40,
            min_num_nodes=4,
            shuffle=True,
            test_size=0.3,
            embed_dim=64,
            hidden_dim=256,
            num_layers=2,
            temperature=0.8,
            force_teacher=0.9,
            sample_interval=5,
            num_intermediate_samples=50,
            num_samples=1000,
            max_epochs=20,
            scheduler_class="StepLR",
            scheduler_params={"step_size": 50, "gamma": 0.5},
            optimizer_class="Adam",
            optimizer_params={'lr': 0.001},
            device="cpu"
        )


class BaselineConfig(BaseConfigWithSerializer):
    def set_defaults(self):
        BaselineConfig._params.update(
            name="ER",
            metric="degree",
            max_num_nodes=40,
            min_num_nodes=4,
            test_size=0.3,
            device="cpu"
        )

class GraphRNNConfig(BaseConfigWithSerializer):
    def set_defaults(self):
        GraphRNNConfig._params.update(
            max_num_nodes=40,
            min_num_nodes=4,
            max_num_node=None,  # max number of nodes in a graph
            max_prev_node=None,  # max previous node that looks back
            hidden_size_rnn=64,
            hidden_size_rnn_output=16,  # hidden size for output RNN
            embedding_size_rnn=32,
            embedding_size_rnn_output=8,  # the embedding size for output rnn
            embedding_size_output=32,
            batch_size=32,
            test_batch_size=32,
            test_total_size=1000,
            test_size=0.3,
            num_layers=3,
            num_workers=4,  # num workers to load data, default 4
            batch_ratio=32,
            epochs=1000,  # now one epoch means batch_ratio x batch_size
            epochs_test_start=1000,
            epochs_test=1000,
            epochs_log=1000,
            epochs_save=1000,
            lr=0.003,
            milestones=[400, 1000],
            lr_rate=0.3,
            sample_time=2,   # sample time in each time step, when validating
            device="cpu"
        )