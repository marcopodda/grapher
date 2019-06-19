from config.config import Config, BaselineConfig, GraphRNNConfig
from utils.evaluation import compute_statistics
from .experiment import Experiment, BaselineExperiment, GraphRNNExperiment
from pathlib import Path
from dataset.manager import get_dataset_class

MODEL_NAMES = ["GRAPHER", "GRAPHRNN", "ER_degree", "ER_clustering", "BA_degree", "BA_clustering"]
DATASET_NAMES = ["community", "ego", "ladders", "ENZYMES", "PROTEINS_full"]


def get_exp_class(model_name):
    if model_name == "GRAPHER":
        return Experiment

    if model_name == "GRAPHRNN":
        return GraphRNNExperiment

    return BaselineExperiment


def get_config_class(model_name):
    if model_name == "GRAPHER":
        return Config

    if model_name == "GRAPHRNN":
        return GraphRNNConfig

    return BaselineConfig


class Evaluator:
    def __init__(self):
        self.kl_degrees = []
        self.kl_clusters = []
        self.root = Path("RUNS")
        self.trials = 10

    def evaluate(self):
        results = {}
        for model_name in MODEL_NAMES:
            results[model_name] = {}
            for dataset_name in DATASET_NAMES:
                results[model_name][dataset_name] = {'degree': [], 'clustering': []}
                rundir = self.root / model_name / dataset_name
                try:
                    expdir = list(rundir.glob("*"))[0]
                except IndexError:
                    continue
                
                config_class = get_config_class(model_name)
                config = config_class.from_file(expdir / "config" / "config.yaml")
                
                exp_class = get_exp_class(model_name)
                exp = exp_class.load(expdir)
                
                dataset_class = get_dataset_class(dataset_name)
                dataset = dataset_class(config, self.root, dataset_name)
                
                test_data = dataset.get_data('test')
                for _ in range(self.trials):
                    samples = exp.sample()
                    kld, klc = compute_statistics(test_data, samples)
                    results[model_name][dataset_name]['degree'] = kld
                    results[model_name][dataset_name]['clustering'] = klc

        return results
