from .base import BaseConfigWithSerializer


class Config(BaseConfigWithSerializer):
    def set_defaults(self):
        Config._params.update(
            order="bfs",
            batch_size=32,
            max_num_edges=130,
            min_num_edges=2,
            max_num_nodes=40,
            min_num_nodes=4,
            shuffle=True,
            test_size=0.3,
            embed_dim=32,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3,
            temperature1=0.75,
            temperature2=0.75,
            max_epochs=2000,
            stop_loss=0.0,
            scheduler_class="StepLR",
            scheduler_params={"step_size": 200, "gamma": 0.5},
            optimizer_class="Adam",
            optimizer_params={'lr': 0.001},
            device="gpu"
        )


class GRUConfig(BaseConfigWithSerializer):
    def set_defaults(self):
        Config._params.update(
            batch_size=32,
            max_num_edges=130,
            min_num_edges=2,
            max_num_nodes=40,
            min_num_nodes=4,
            shuffle=True,
            test_size=0.3,
            embed_dim=32,
            hidden_dim=128,
            num_layers=2,
            max_epochs=800,
            dropout=0.3,
            stop_loss=0.1,
            temperature=0.8,
            scheduler_class="StepLR",
            scheduler_params={"step_size": 100, "gamma": 0.5},
            optimizer_class="Adam",
            optimizer_params={'lr': 0.001},
            device="gpu"
        )


class BaselineConfig(BaseConfigWithSerializer):
    def set_defaults(self):
        BaselineConfig._params.update(
            name="ER",
            metric="degree",
            max_num_nodes=40,
            min_num_nodes=4,
            max_num_edges=130,
            min_num_edges=2,
            test_size=0.3,
            device="gpu"
        )


class GraphRNNConfig(BaseConfigWithSerializer):
    def set_defaults(self):
        GraphRNNConfig._params.update(
            max_num_nodes=40,
            min_num_nodes=4,
            max_num_edges=130,
            min_num_edges=2,
            max_num_node=None,  # max number of nodes in a graph
            max_prev_node=None,  # max previous node that looks back
            hidden_size_rnn=64,
            hidden_size_rnn_output=16,  # hidden size for output RNN
            embedding_size_rnn=32,
            embedding_size_rnn_output=8,  # the embedding size for output rnn
            embedding_size_output=32,
            batch_size=32,
            test_size=0.3,
            num_layers=3,
            num_workers=4,  # num workers to load data, default 4
            batch_ratio=32,
            epochs=2000,  # now one epoch means batch_ratio x batch_size
            lr=0.003,
            milestones=[400, 1000],
            lr_rate=0.3,
            device="gpu"
        )


def get_config_class(model_name):
    if model_name == "GRAPHER":
        return Config

    if model_name == "GRAPHRNN":
        return GraphRNNConfig

    if model_name == "GRU":
        return GRUConfig

    if model_name in ["ER", "BA"]:
        return BaselineConfig

    return Config