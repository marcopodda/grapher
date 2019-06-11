from .base import BaseConfigWithSerializer


class Config(BaseConfigWithSerializer):
    def set_defaults(self):
        Config._params.update({
            "bfs_order": True,
            "batch_size": 16,
            "max_num_nodes": 40,
            "min_num_nodes": 4,
            "shuffle": True,
            "test_size": 0.1,
            "embed_dim": 64,
            "hidden_dim": 256,
            "num_layers": 2,
            "temperature": 0.8,
            "force_teacher": 0.9,
            "sample_interval": 5,
            "num_intermediate_samples": 50,
            "num_samples": 1000,
            "max_epochs": 20,
            "scheduler_class": "StepLR",
            "scheduler_params": {"step_size": 50, "gamma": 0.5},
            "optimizer_class": "Adam",
            "optimizer_params": {'lr': 0.001}
        })
