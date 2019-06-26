import torch
import numpy as np
from pathlib import Path

from config import get_config_class
from learner import get_exp_class
from dataset import get_dataset_class

from utils.evaluation import degree_kl, clustering_kl
from utils.serializer import save_yaml

MODEL_NAMES = ["GRAPHER", "GRAPHRNN", "ER_degree", "ER_clustering", "BA_degree", "BA_clustering"]
DATASET_NAMES = ["community", "ego", "ladders", "ENZYMES", "PROTEINS_full"]


def pad_and_add(v1, v2):
    if v1 is None:
        return v2

    maxdim = max(len(v1), len(v2))
    newvec = np.zeros((maxdim,))
    newvec[:len(v1)] = v1
    newvec[:len(v2)] = v2
    return newvec.tolist()


class Results:
    def __init__(self, model_names, dataset_names):
        self.results = {}
        self.model_names = model_names
        self.dataset_names = dataset_names

        for model_key in self.model_names:
            self.results[model_key] = {}
            for dataset_key in self.dataset_names:
                self.results[model_key][dataset_key] = {
                    'degree': [],
                    'degree_mean': None,
                    'degree_std': None,
                    'degree_count_data': None,
                    'degree_count_samples': None,
                    'clustering': [],
                    'clustering_mean': None,
                    'clustering_std': None,
                    'clustering_count_data': None,
                    'clustering_count_samples': None,
                }

    def set_perf(self, model_key, dataset_key, perf_key, value):
        self.results[model_key][dataset_key][perf_key].append(float(value))

    def get_perf(self, model_key, dataset_key, perf_key):
        return self.results[model_key][dataset_key][perf_key]

    def get_count(self, model_key, dataset_key, perf_key, data_key):
        key = f"{perf_key}_count_{data_key}"
        return self.results[model_key][dataset_key][key]

    def set_count(self, model_key, dataset_key, perf_key, data_key, new_value):
        key = f"{perf_key}_count_{data_key}"
        value = self.get_count(model_key, dataset_key, perf_key, data_key)
        self.results[model_key][dataset_key][key] = pad_and_add(value, new_value)

    def set_perf_stat(self, model_key, dataset_key, perf_key):
        perf = self.get_perf(model_key, dataset_key, perf_key)
        self.results[model_key][dataset_key][f"{perf_key}_mean"] = float(np.mean(perf))
        self.results[model_key][dataset_key][f"{perf_key}_std"] = float(np.std(perf))
        self.results[model_key][dataset_key][perf_key] = self.results[model_key][dataset_key][perf_key]

    def set_count_stat(self, model_key, dataset_key, perf_key, num_trials):
        key = f"{perf_key}_count_data"
        for i, _ in enumerate(self.results[model_key][dataset_key][key]):
            self.results[model_key][dataset_key][key][i] /= num_trials

        key = f"{perf_key}_count_samples"
        for i, _ in enumerate(self.results[model_key][dataset_key][key]):
            self.results[model_key][dataset_key][key][i] /= num_trials


class Evaluator:
    def __init__(self):
        self.root = Path("RUNS")
        self.num_trials = 10
        self.results = Results(MODEL_NAMES, DATASET_NAMES)

    def _load_experiment(self, model_name, dataset_name):
        rundir = self.root / model_name / dataset_name
        expdir = list(rundir.glob("*"))
        assert len(expdir) > 0
        exp_class = get_exp_class(model_name)
        exp = exp_class.load(expdir[0])
        return exp

    def _load_dataset(self, dataset_name, model_name, exp):
        config_class = get_config_class(model_name)
        config = config_class.from_file(exp.root / "config" / "config.yaml")
        config.update(temperature=0.1)
        dataset_class = get_dataset_class(dataset_name)
        dataset = dataset_class(config, exp.root, dataset_name)
        return dataset

    def evaluate(self):
        for model_name in self.results.model_names:
            print(model_name)
            for dataset_name in self.results.dataset_names:
                print("  ", dataset_name)
                exp = self._load_experiment(model_name, dataset_name)
                dataset = self._load_dataset(dataset_name, model_name, exp)
                test_data = dataset.get_data('test')

                for i, trial in enumerate(range(self.num_trials)):
                    samples = exp.sample(num_samples=len(test_data))
                    torch.save(samples, exp.root / "samples" / f"samples_{i}.pt")

                    kld, d_count_data, d_count_samples = degree_kl(test_data, samples)
                    self.results.set_perf(model_name, dataset_name, 'degree', kld)
                    self.results.set_count(model_name, dataset_name, 'degree', 'data', d_count_data)
                    self.results.set_count(model_name, dataset_name, 'degree', 'samples', d_count_samples)

                    klc, c_count_data, c_count_samples = clustering_kl(test_data, samples)
                    self.results.set_perf(model_name, dataset_name, 'clustering', klc)
                    self.results.set_count(model_name, dataset_name, 'clustering', 'data', c_count_data)
                    self.results.set_count(model_name, dataset_name, 'clustering', 'samples', c_count_samples)

                self.results.set_perf_stat(model_name, dataset_name, 'degree')
                self.results.set_perf_stat(model_name, dataset_name, 'clustering')
                self.results.set_count_stat(model_name, dataset_name, 'degree', self.num_trials)
                self.results.set_count_stat(model_name, dataset_name, 'clustering', self.num_trials)

        save_yaml(self.results.results, self.root / "results.yaml")
        return self.results


class OrderEvaluator:
    def __init__(self):
        self.root = Path("RUNS") / "ORDER"
        self.num_trials = 10
        self.results = Results(["bfs", "random", "smiles"], DATASET_NAMES)

    def _load_experiment(self, model_name, dataset_name):
        rundir = self.root / model_name / dataset_name
        expdir = list(rundir.glob("*"))
        assert len(expdir) > 0
        exp_class = get_exp_class(model_name)
        exp = exp_class.load(expdir[0])
        return exp

    def _load_dataset(self, dataset_name, model_name, exp):
        config_class = get_config_class(model_name)
        config = config_class.from_file(exp.root / "config" / "config.yaml")
        dataset_class = get_dataset_class(dataset_name)
        dataset = dataset_class(config, exp.root, dataset_name)
        return dataset

    def evaluate(self):
        for model_name in self.results.model_names:
            print(model_name)
            for dataset_name in self.results.dataset_names:

                if model_name == "smiles" and dataset_name not in ["ENZYMES", "PROTEINS_full"]:
                    continue
                print("  ", dataset_name)

                exp = self._load_experiment(model_name, dataset_name)
                dataset = self._load_dataset(dataset_name, model_name, exp)
                test_data = dataset.get_data('test')

                for i, trial in enumerate(range(self.num_trials)):
                    samples = exp.sample(num_samples=len(test_data))
                    torch.save(samples, exp.root / "samples" / f"samples_{i}.pt")

                    kld, d_count_data, d_count_samples = degree_kl(test_data, samples)
                    self.results.set_perf(model_name, dataset_name, 'degree', kld)
                    self.results.set_count(model_name, dataset_name, 'degree', 'data', d_count_data)
                    self.results.set_count(model_name, dataset_name, 'degree', 'samples', d_count_samples)

                    klc, c_count_data, c_count_samples = clustering_kl(test_data, samples)
                    self.results.set_perf(model_name, dataset_name, 'clustering', klc)
                    self.results.set_count(model_name, dataset_name, 'clustering', 'data', c_count_data)
                    self.results.set_count(model_name, dataset_name, 'clustering', 'samples', c_count_samples)

                self.results.set_perf_stat(model_name, dataset_name, 'degree')
                self.results.set_perf_stat(model_name, dataset_name, 'clustering')
                self.results.set_count_stat(model_name, dataset_name, 'degree', self.num_trials)
                self.results.set_count_stat(model_name, dataset_name, 'clustering', self.num_trials)

        save_yaml(self.results.results, self.root / "results.yaml")
        return self.results
