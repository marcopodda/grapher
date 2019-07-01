import torch
import numpy as np
from pathlib import Path

from config import get_config_class
from learner import get_exp_class
from dataset import get_dataset_class

from utils import evaluation
from utils.constants import MODEL_NAMES, DATASET_NAMES, ORDER_NAMES
from utils.serializer import save_yaml


def pad_and_add(v1, v2):
    if v1 is None:
        return v2

    maxdim = max(len(v1), len(v2))
    newvec = np.zeros((maxdim,))
    newvec[:len(v1)] = v1
    newvec[:len(v2)] = v2
    return newvec.tolist()


class Results:
    def __init__(self, metric, model_names, dataset_names):
        self.results = {}
        self.metric = metric
        self.model_names = model_names
        self.dataset_names = dataset_names

        for model_key in self.model_names:
            self.results[model_key] = {}
            for dataset_key in self.dataset_names:
                self.results[model_key][dataset_key] = {
                    'scores': [],
                    'mean': None,
                    'std': None,
                    'count_data': None,
                    'count_samples': None,
                    'novelty_score': None,
                    'uniqueness_score': None,
                }

    def set_perf(self, model_key, dataset_key, value):
        self.results[model_key][dataset_key]["scores"].append(float(value))

    def get_perf(self, model_key, dataset_key):
        return self.results[model_key][dataset_key]["scores"]

    def get_count(self, model_key, dataset_key, data_key):
        key = f"count_{data_key}"
        return self.results[model_key][dataset_key][key]

    def set_count(self, model_key, dataset_key, data_key, new_value):
        key = f"count_{data_key}"
        value = self.get_count(model_key, dataset_key, data_key)
        self.results[model_key][dataset_key][key] = pad_and_add(value, new_value)

    def set_perf_stat(self, model_key, dataset_key):
        perf = self.get_perf(model_key, dataset_key)
        self.results[model_key][dataset_key]["mean"] = float(np.mean(perf))
        self.results[model_key][dataset_key]["std"] = float(np.std(perf))
        self.results[model_key][dataset_key]["scores"] = self.results[model_key][dataset_key]["scores"]

    def set_count_stat(self, model_key, dataset_key, num_trials):
        key = "count_data"
        for i, _ in enumerate(self.results[model_key][dataset_key][key]):
            self.results[model_key][dataset_key][key][i] /= num_trials

        key = "count_samples"
        for i, _ in enumerate(self.results[model_key][dataset_key][key]):
            self.results[model_key][dataset_key][key][i] /= num_trials

    def set_novelty_score(self, model_key, dataset_key, score):
        self.results[model_key][dataset_key]['novelty_score'] = score

    def set_uniqueness_score(self, model_key, dataset_key, score):
        self.results[model_key][dataset_key]['uniqueness_score'] = score


class EvaluatorBase:
    root = Path("RUNS")
    num_samples = 10000

    def _eval(self, model_name, dataset_name, test_data, samples):
        func = getattr(evaluation, f'{self.metric}_kl')
        kl, count_data, count_samples = func(test_data, samples)
        self.results.set_perf(model_name, dataset_name, kl)
        self.results.set_count(model_name, dataset_name, 'data', count_data)
        self.results.set_count(model_name, dataset_name, 'samples', count_samples)

    def _calc_mean(self, model_name, dataset_name):
        self.results.set_perf_stat(model_name, dataset_name)
        self.results.set_count_stat(model_name, dataset_name, self.num_trials)

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


def get_novelty_score(train, samples):
    def process(G):
        mapping = {n:n+3 for n in G.nodes()}
        return sorted(G.edges())
    train = [process(G) for G in train]
    novel = [e for e in samples if e not in train]
    return len(novel) / len(samples)


def get_uniqueness_score(samples):
    unique = set([tuple(e) for e in samples])
    return len(unique) / len(samples)


class Evaluator(EvaluatorBase):
    def __init__(self, metric):
        self.metric = metric
        self.results = Results(metric, MODEL_NAMES, DATASET_NAMES)

    def evaluate(self):
        for model_name in self.results.model_names:
            print(model_name)
            for dataset_name in self.results.dataset_names:
                print("  ", dataset_name)
                exp = self._load_experiment(model_name, dataset_name)
                dataset = self._load_dataset(dataset_name, model_name, exp)

                train_data = dataset.get_data('train')
                test_data = dataset.get_data('test')

                if not (exp.root / "samples" / f"samples.pt").exists():
                    samples = exp.sample(num_samples=self.num_samples)
                    torch.save(samples, exp.root / "samples" / f"samples.pt")
                samples = torch.load(exp.root / "samples" / f"samples.pt")

                # novelty_score = get_novelty_score(train_data, samples)
                # self.results.set_novelty_score(model_name, dataset_name, novelty_score)
                # uniqueness_score = get_uniqueness_score(train_data, samples)
                # self.results.set_uniqueness_score(model_name, dataset_name, uniqueness_score)
                # self._eval(model_name, dataset_name, test_data, samples)
                # self._calc_mean(model_name, dataset_name)

        # save_yaml(self.results.results, self.root / f"results_{self.metric}.yaml")
        return self.results


class OrderEvaluator(EvaluatorBase):
    root = Path("RUNS") / "ORDER"

    def __init__(self, metric):
        self.metric = metric
        self.results = Results(metric, ORDER_NAMES, DATASET_NAMES)

    def evaluate(self):
        for model_name in self.results.model_names:
            print(model_name)
            for dataset_name in self.results.dataset_names:
                if model_name == "smiles" and dataset_name not in ["ENZYMES", "PROTEINS_full"]:
                    continue
                print("  ", dataset_name)

                exp = self._load_experiment(model_name, dataset_name)
                dataset = self._load_dataset(dataset_name, model_name, exp)
                train_data = dataset.get_data('train')
                test_data = dataset.get_data('test')

                if not (exp.root / "samples" / f"samples.pt").exists():
                    samples = exp.sample(num_samples=self.num_samples)
                    torch.save(samples, exp.root / "samples" / f"samples.pt")
                samples = torch.load(exp.root / "samples" / f"samples.pt")

                # novelty_score = get_novelty_score(train_data, samples)
                # self.results.set_novelty_score(model_name, dataset_name, novelty_score)
                # uniqueness_score = get_uniqueness_score(train_data, samples)
                # self.results.set_uniqueness_score(model_name, dataset_name, uniqueness_score)
                # self._eval(model_name, dataset_name, test_data, samples)
                # self._calc_mean(model_name, dataset_name)

        # save_yaml(self.results.results, self.root / f"results_{self.metric}.yaml")
        return self.results
