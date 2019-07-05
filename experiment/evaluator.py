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
from utils.serializer import save_yaml, load_yaml


def pad_and_add(v1, v2):
    if v1 is None:
        return v2

    maxdim = max(len(v1), len(v2))
    newvec = np.zeros((maxdim,))
    newvec[:len(v1)] = v1
    newvec[:len(v2)] = v2
    return newvec


class Metric:
    @classmethod
    def load(cls, datadict):
        metric = cls()
        metric.scores = datadict['scores']
        metric.mean = datadict['mean']
        metric.std = datadict['std']
        metric.data_hist = datadict['data_hist']
        metric.samples_hist = datadict['samples_hist']
        return metric

    def __init__(self):
        self.scores = []
        self.mean = None
        self.std = None
        self.data_hist = None
        self.samples_hist = None

    @property
    def is_empty(self):
        return self.scores == []

    def get_score_func(self):
        raise NotImplementedError

    def update(self, test_data, samples):
        kld, data_hist, samples_hist = evaluation.kl_divergence(test_data, samples, self.name)
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
    @classmethod
    def load(cls, model_name, dataset_name, path):
        resultdict = load_yaml(path)
        r = cls(model_name, dataset_name)
        r.degree = DegreeDistribution.load(resultdict['degree'])
        r.clustering = ClusteringCoefficient.load(resultdict['clustering'])
        r.graphlet = GraphletCount.load(resultdict['graphlet'])
        r.novelty1000 = resultdict['novelty1000']
        r.novelty10000 = resultdict['novelty10000']
        r.uniqueness1000 = resultdict['uniqueness1000']
        r.uniqueness10000 = resultdict['uniqueness10000']
        r.time1000 = resultdict['time1000']
        r.time10000 = resultdict['time10000']
        return r

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

    @property
    def uniqueness_not_calculated(self):
        return self.uniqueness1000 is None and self.uniqueness10000 is None

    @property
    def novelty_not_calculated(self):
        return self.novelty1000 is None and self.novelty10000 is None

    def asdict(self):
        data = self.__dict__
        data['degree'] = self.degree.asdict()
        data['clustering'] = self.clustering.asdict()
        data['graphlet'] = self.graphlet.asdict()
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
        self.num_samples = [1000, 10000]
        self.num_trials = 10

    def evaluate(self):
        for dataset_name in DATASET_NAMES:
            print(dataset_name)
            exp = load_experiment(self.root, self.model_name, dataset_name)
            dataset = load_dataset(dataset_name, self.model_name, exp)

            if not (exp.root / "results" / dataset_name).exists():
                result = Result(self.model_name, dataset_name)
            else:
                result = load_yaml(exp.root / "results" / dataset_name)

            if result.novelty_not_calculated:
                self.evaluate_novelty(result, exp, dataset)

            if result.uniqueness_not_calculated:
                self.evaluate_uniqueness(result, exp, dataset)

            if result.degree.is_empty:
                self.evaluate_kl(result, exp, dataset, 'degree')

            if result.clustering.is_empty:
                self.evaluate_kl(result, exp, dataset, 'clustering')

            # if result.graphlet.is_empty:
            #     self.evaluate_kl(result, exp, dataset, 'graphlet')

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

    def evaluate_kl(self, result, exp, dataset, metric):
        test_data = dataset.get_data('test')
        for trial in range(self.num_trials):
            _, samples = self._sample_or_get_samples(result, exp, len(test_data), trial)
            result.update_metric(metric, test_data, samples)
        result.finalize_metric(metric, self.num_trials)


class Evaluator(EvaluatorBase):
    root = Path("RUNS")


class OrderEvaluator(EvaluatorBase):
    root = Path("RUNS") / "ORDER"
