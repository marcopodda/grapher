import time
import torch
import numpy as np
import networkx as nx
from pathlib import Path

from config import get_config_class
from dataset import get_dataset_class, load_dataset
from dataset.graph import GraphList
from .experiment import load_experiment

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
    return newvec


class Metric:
    def __init__(self):
        self.scores = []
        self.mean = None
        self.std = None
        self.data_hist = None
        self.samples_hist = None

    def get_score_func(self):
        raise NotImplementedError

    def update(self, test_data, samples):
        kld, data_hist, samples_hist = evaluation.kl_divergence(test_data, samples, self.name)
        print(data_hist, samples_hist, kld)
        self.scores.append(float(kld))
        self.data_hist = pad_and_add(self.data_hist, data_hist)
        self.samples_hist = pad_and_add(self.samples_hist, samples_hist)

    def finalize(self, num_trials):
        self.mean = float(np.mean(self.scores))
        self.std = float(np.std(self.scores))
        self.data_hist = [float(x) for x in self.data_hist / num_trials]
        self.samples_hist = [float(x) for x in self.samples_hist / num_trials]

    def asdict(self):
        return self.__dict__


class DegreeDistribution(Metric):
    name = "degree"


class ClusteringCoefficient(Metric):
    name = "clustering"


class GraphletCount(Metric):
    name = "graphlet"


class Result:
    def __init__(self, model_name, dataset_name):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.degree = DegreeDistribution()
        self.clustering = ClusteringCoefficient()
        self.graphlet = GraphletCount()
        self.novelty1000 = None
        self.novelty10000 = None
        self.uniqueness1000 = None
        self.uniqueness10000 = None
        self.time1000 = None
        self.time10000 = None

    def asdict(self):
        data = self.__dict__
        data['degree'] = data.degree.asdict()
        data['clustering'] = data.clustering.asdict()
        data['graphlet'] = data.graphlet.asdict()
        return data

    def save(self, path):
        save_yaml(self.asdict(), path / f"{self.dataset_name}.yaml")

    def update(self, name, value):
        setattr(self, name, value)

    def update_time(self, num_samples, time_elapsed):
        if time_elapsed is not None:
            setattr(self, f"time{num_samples}", time_elapsed)

    def update_metric(self, name, test_data, samples):
        metric = getattr(self, name)
        metric.update(test_data, samples)

    def finalize_metric(self, name, num_trials):
        metric = getattr(self, name)
        metric.finalize(num_trials)


class EvaluatorBase:
    def __init__(self, model_name):
        self.model_name = model_name
        self.num_samples = [10, 20]
        self.num_trials = 3

    def evaluate(self):
        for dataset_name in DATASET_NAMES:
            result = Result(self.model_name, dataset_name)
            exp = load_experiment(self.root, self.model_name, dataset_name)
            dataset = load_dataset(dataset_name, self.model_name, exp)

            self.evaluate_novelty(result, exp, dataset)
            self.evaluate_uniqueness(result, exp, dataset)
            self.evaluate_samples(result, exp, dataset)

            result.save(exp.root / "results")

    def _sample_or_get_samples(self, result, exp, num_samples, trial=None):
        time_elapsed = None

        if trial is not None:
            filename = f"samples_{num_samples}_{trial}.pt"
        else:
            filename = f"samples_{num_samples}.pt"

        if not (exp.root / "samples" /filename).exists():
            start = time.time()
            samples = exp.sample(num_samples=num_samples)
            time_elapsed = time.time() - start
            torch.save(samples, exp.root / "samples" /filename)
        return time_elapsed, torch.load(exp.root / "samples" /filename)

    def evaluate_novelty(self, result, exp, dataset):
        train_data = dataset.get_data('train')
        for num_samples in self.num_samples:
            time_elapsed, samples = self._sample_or_get_samples(result, exp, num_samples)
            result.update(f'novelty{num_samples}', evaluation.novelty(train_data, samples))
            result.update_time(num_samples, time_elapsed)

    def evaluate_uniqueness(self, result, exp, dataset):
        for num_samples in self.num_samples:
            time_elapsed, samples = self._sample_or_get_samples(result, exp, num_samples)
            result.update(f'uniqueness{num_samples}', evaluation.uniqueness(samples))
            result.update_time(num_samples, time_elapsed)

    def evaluate_samples(self, result, exp, dataset):
        test_data = dataset.get_data('test')
        for trial in range(self.num_trials):
            _, samples = self._sample_or_get_samples(result, exp, len(test_data), trial)
            result.update_metric('degree', test_data, samples)
            result.update_metric('clustering', test_data, samples)
            # result.update_metric('graphlet', test_data, samples)
        result.finalize_metric('degree', self.num_trials)
        result.finalize_metric('clustering', self.num_trials)
        # result.finalize_metric('graphlet', self.num_trials)

class Evaluator(EvaluatorBase):
    root = Path("RUNS")


class OrderEvaluator(EvaluatorBase):
    root = Path("RUNS") / "ORDER"