from .config import BaselineConfig, Config, GraphRNNConfig, GRUConfig


def get_config_class(model_name):
    if model_name == "GRAPHER":
        return Config

    if model_name == "GRU":
        return GRUConfig

    if model_name == "GRAPHRNN":
        return GraphRNNConfig

    if model_name in ["ER", "BA"]:
        return BaselineConfig

    return Config
