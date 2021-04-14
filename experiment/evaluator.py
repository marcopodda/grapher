import time
import torch
import numpy as np
import networkx as nx
from pathlib import Path

from dataset import load_dataset
from .experiment import load_experiment

from utils import evaluation
from utils.constants import DATASET_NAMES
from utils.serializer import save_yaml, load_yaml


def pad_and_add(v1, v2):
    if v1 is None:
        return v2

    maxdim = max(len(v1), len(v2))
    new_v1 = np.zeros((maxdim,))
    new_v2 = np.zeros((maxdim,))
    new_v1[:len(v1)] = v1
    new_v2[:len(v2)] = v2
    return new_v1 + new_v2


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
    def is_computed(self):
        return self.scores != []

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


class OrbitCount(Metric):
    name = "orbit"


class BetweennessCentrality(Metric):
    name = "betweenness"


class Result:
    @classmethod
    def load(cls, model_name, dataset_name, path):
        resultdict = load_yaml(path)
        result = cls(model_name, dataset_name)
        result.degree = DegreeDistribution.load(resultdict.pop('degree'))
        result.clustering = ClusteringCoefficient.load(resultdict.pop('clustering'))
        result.orbit = OrbitCount.load(resultdict.pop('orbit'))
        result.betweenness = OrbitCount.load(resultdict.pop('betweenness'))
        for key in resultdict:
            setattr(result, key, resultdict[key])
        return result

    def __init__(self, model_name, dataset_name):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.degree = DegreeDistribution()
        self.clustering = ClusteringCoefficient()
        self.orbit = OrbitCount()
        self.betweenness = BetweennessCentrality()

    @property
    def uniqueness_not_calculated(self):
        return not hasattr(self, 'uniqueness1000')

    @property
    def novelty_not_calculated(self):
        return not hasattr(self, 'novelty1000')

    def asdict(self):
        data = self.__dict__
        data['degree'] = self.degree.asdict()
        data['clustering'] = self.clustering.asdict()
        data['orbit'] = self.orbit.asdict()
        data['betweenness'] = self.betweenness.asdict()
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

    def clean_orbit(self):
        self.orbit.scores = []
        self.orbit.data_hist = None
        self.orbit.samples_hist = None
        self.orbit.mean = None
        self.orbit.std = None

    def clean_degree(self):
        self.degree.scores = []
        self.degree.data_hist = None
        self.degree.samples_hist = None
        self.degree.mean = None
        self.degree.std = None

    def clean_clustering(self):
        self.clustering.scores = []
        self.clustering.data_hist = None
        self.clustering.samples_hist = None
        self.clustering.mean = None
        self.clustering.std = None

    def clean_betweenness(self):
        self.betweenness.scores = []
        self.betweenness.data_hist = None
        self.betweenness.samples_hist = None
        self.betweenness.mean = None
        self.betweenness.std = None


class EvaluatorBase:
    def __init__(self, model_name):
        self.model_name = model_name
        self.num_samples = [1000, 5000]
        self.num_trials = 10
        self.fast = model_name == "GRAPHER"

    def novelty_not_calculated(self, result):
        return result.novelty_not_calculated

    def uniqueness_not_calculated(self, result):
        return result.uniqueness_not_calculated

    def evaluate(self):
        for dataset_name in DATASET_NAMES:
            if self.model_name == "smiles" and dataset_name not in ["PROTEINS_full", "ENZYMES"]:
                continue
            print(dataset_name)
            exp = load_experiment(self.root, self.model_name, dataset_name)
            dataset = load_dataset(dataset_name, self.model_name, exp)

            path = exp.root / "results" / f"{dataset_name}.yaml"
            if not path.exists():
                result = Result(self.model_name, dataset_name)
            else:
                result = Result.load(self.model_name, dataset_name, path)

            if self.novelty_not_calculated(result):
                self.evaluate_novelty(result, exp, dataset)

            if self.uniqueness_not_calculated(result):
                self.evaluate_uniqueness(result, exp, dataset)

            if not result.degree.is_computed:
                self.evaluate_kl(result, exp, dataset, 'degree')

            if not result.clustering.is_computed:
                self.evaluate_kl(result, exp, dataset, 'clustering')

            if not result.orbit.is_computed:
                self.evaluate_kl(result, exp, dataset, 'orbit')

            if not result.betweenness.is_computed:
                self.evaluate_kl(result, exp, dataset, 'betweenness')

            result.save(exp.root / "results")

    def _sample_or_get_samples_kl(self, result, exp, num_samples, trial):
        filename = f"samples_{num_samples}_{trial}.pt"

        if not (exp.root / "samples" / filename).exists():
            samples = exp.sample(num_samples=num_samples)
            torch.save(samples, exp.root / "samples" /filename)
        samples = torch.load(exp.root / "samples" / filename)
        return [G for G in samples if G.number_of_nodes() > 0]

    def _sample_or_get_samples_metric(self, result, exp, num_samples, trial=None):
        time_elapsed = None
        filename = f"samples_{num_samples}.pt"

        if not (exp.root / "samples" /filename).exists():
            start = time.time()
            samples = exp.sample(num_samples=num_samples)
            time_elapsed = time.time() - start
            torch.save(samples, exp.root / "samples" /filename)

        samples = torch.load(exp.root / "samples" / filename)
        return time_elapsed, [G for G in samples if G.number_of_nodes() > 0]

    def evaluate_novelty(self, result, exp, dataset):
        train_data = dataset.get_data('train')
        for num_samples in self.num_samples:
            time_elapsed, samples = self._sample_or_get_samples_metric(result, exp, num_samples)
            novelty, _ = evaluation.novelty(train_data, samples, self.fast)
            result.update(f'novelty{num_samples}', novelty)
            result.update_time(num_samples, time_elapsed)

    def evaluate_uniqueness(self, result, exp, dataset):
        for num_samples in self.num_samples:
            time_elapsed, samples = self._sample_or_get_samples_metric(result, exp, num_samples)
            uniqueness, _ = evaluation.uniqueness(samples, self.fast)
            result.update(f'uniqueness{num_samples}', uniqueness)
            result.update_time(num_samples, time_elapsed)

    def evaluate_kl(self, result, exp, dataset, metric_name):
        test_data = dataset.get_data('test')

        for trial in range(self.num_trials):
            samples = self._sample_or_get_samples_kl(result, exp, len(test_data), trial)
            result.update_metric(metric_name, test_data, samples)
        result.finalize_metric(metric_name, self.num_trials)


class Evaluator(EvaluatorBase):
    root = Path("RUNS")


class OrderEvaluator(EvaluatorBase):
    root = Path("RUNS") / "ORDER"

    def __init__(self, model_name):
        super().__init__(model_name)
        self.fast = True

    def novelty_not_calculated(self, result):
        return False

    def uniqueness_not_calculated(self, result):
        return False