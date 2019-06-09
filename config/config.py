from .base import BaseConfig


class Config(BaseConfig):
    def set_defaults(self):
        Config._params.update({
            "batch_size": 10,
            "max_num_nodes": 20,
            "max_num_edges": 20,
            "shuffle": True,
            "test_size": 0.1
        })