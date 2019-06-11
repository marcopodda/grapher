from .base import BaseConfig


class Config(BaseConfig):
    def set_defaults(self):
        Config._params.update({
            "bfs_order": True,
            "batch_size": 16,
            "max_num_nodes": 40,
            "shuffle": True,
            "test_size": 0.1,
            "embed_dim": 64,
            "hidden_dim": 256,
            "num_layers": 2,
            "temperature": 0.5,
            "force_teacher": 0.9,
            "max_epochs": 150,
            "scheduler_class": "StepLR",
            "scheduler_params": {"step_size": 50, "gamma": 0.5},
            "optimizer_class": "Adam",
            "optimizer_params": {'lr': 0.001}
        })