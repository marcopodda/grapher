from .config import BaselineConfig, Config, GraphRNNConfig


def get_config_class(model_name):
    if model_name in ["GRAPHER"]:
        return Config

    if model_name == "GRAPHRNN":
        return GraphRNNConfig

    if model_name in ["ER_degree", "ER_clustering", "BA_degree", "BA_clustering"]:
        return BaselineConfig

    return Config
