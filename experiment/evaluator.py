import itertools
import time
from joblib.parallel import Parallel, delayed
import torch
import numpy as np
import networkx as nx
from pathlib import Path
from functools import partial
from scipy.stats import entropy

from dataset import load_dataset
from .experiment import load_experiment
from .eval import (
    degree_dist,
    clustering_dist,
    nspdk_dist,
    orbit_dist,
    betweenness_dist,
    patch,
    random_sample,
    novelty,
    uniqueness,
    normalize)

from utils import mmd
from utils.constants import DATASET_NAMES, ORDER_NAMES


EPS = 1e-8


METRICS = {
    "degree": degree_dist,
    "clustering": clustering_dist,
    "orbit": orbit_dist,
    "betweenness": betweenness_dist,
    "nspdk": nspdk_dist
}


def calc_num_nodes(graphs):
    return max([G.number_of_nodes() for G in graphs])


class EvaluatorBase:
    requires_quantitative = True
    def __init__(self, model_name):
        self.model_name = model_name
        self.num_samples = 5000
        self.num_samples_small = 1000
        self.num_trials = 5
        self.fast = model_name == "GRAPHER" or model_name in ORDER_NAMES

    def novelty_not_calculated(self, result):
        return result.novelty_not_calculated

    def uniqueness_not_calculated(self, result):
        return result.uniqueness_not_calculated

    def evaluate(self):
        for dataset_name in DATASET_NAMES:
            if self.model_name == "smiles" and dataset_name not in ["PROTEINS_full", "ENZYMES"]:
                continue
            print(f"--------------Evaluating {self.model_name} on {dataset_name}--------------")
            exp = load_experiment(self.root, self.model_name, dataset_name)
            dataset = load_dataset(dataset_name, self.model_name, exp)

            path = exp.root / "results" / f"results.pt"
            samples = self.get_samples(exp)

            if not path.exists():
                result = {}
                if self.requires_quantitative:
                    print("\tCalculating novelty...")
                    novelty_small, novelty_large = self.evaluate_novelty(dataset, samples)
                    print("\tCalculating uniqueness...")
                    uniqueness_small, uniqueness_large = self.evaluate_uniqueness(samples)
                    result.update(**{
                        f"novelty{self.num_samples}": novelty_large,
                        f"uniqueness{self.num_samples}": uniqueness_large,
                        f"novelty{self.num_samples_small}": novelty_small,
                        f"uniqueness{self.num_samples_small}": uniqueness_small,
                    })
                print("\tCalculating degree distribution...")
                degree = self.evaluate_metric('degree', dataset, samples)
                print("\tCalculating clustering coefficient...")
                clustering = self.evaluate_metric('clustering', dataset, samples)
                print("\tCalculating orbit counts...")
                orbit = self.evaluate_metric('orbit', dataset, samples)
                print("\tCalculating betweenness centrality...")
                betweenness = self.evaluate_metric('betweenness', dataset, samples)
                print("\tCalculating NSPDK...")
                nspdk = self.evaluate_metric('nspdk', dataset, samples)
                result.update(**{
                    "degree": degree,
                    "clustering": clustering,
                    "orbit": orbit,
                    "betweenness": betweenness,
                    "nspdk": nspdk
                })
                torch.save(result, path)
                print("\tDone.")
            else:
                if self.requires_quantitative:
                    result = torch.load(path)
                    print("\tCalculating novelty...")
                    novelty_small, novelty_large = self.evaluate_novelty(dataset, samples)
                    print("\tCalculating uniqueness...")
                    uniqueness_small, uniqueness_large = self.evaluate_uniqueness(samples)
                    print("\tCalculating orbit counts...")
                    orbit = self.evaluate_metric('orbit', dataset, samples)
                    result.update(**{
                        f"novelty{self.num_samples}": novelty_large,
                        f"uniqueness{self.num_samples}": uniqueness_large,
                        f"novelty{self.num_samples_small}": novelty_small,
                        f"uniqueness{self.num_samples_small}": uniqueness_small,
                        "orbit": orbit
                    })
                    result = torch.save(result, path)
                print("\tAlready evaluated, skipping.")

    def get_samples(self, exp):
        time_elapsed = None
        filename = f"samples.pt"

        if not (exp.root / "samples" / filename).exists():
            print("\tGetting samples...", end=" ")
            start = time.time()
            P = Parallel(n_jobs=48, verbose=0)
            samples = P(delayed(exp.sample)(1) for _ in range(self.num_samples))
            # samples = exp.sample(num_samples=self.num_samples)
            time_elapsed = time.time() - start
            with open(exp.root / "samples" / "elapsed.txt", "w") as f:
                print(time_elapsed, file=f)
            samples = list(itertools.chain.from_iterable(samples))
            torch.save(samples, exp.root / "samples" / filename)
            print("Done.")
        else:
            print("\tSamples ready.")

        samples = torch.load(exp.root / "samples" / filename)
        return [G for G in samples if G.number_of_nodes() > 1 and G.number_of_edges() > 0]

    def evaluate_novelty(self, dataset, samples):
        train_data = dataset.get_data('train')
        min_num_samples = min(len(samples), self.num_samples_small)
        indices = np.random.choice(len(samples), min_num_samples, replace=False)
        samples_small = [samples[i] for i in indices]
        novelty_small = novelty(train_data, samples_small, fast=self.fast)
        novelty_large = novelty(train_data, samples, fast=self.fast)
        return novelty_small, novelty_large

    def evaluate_uniqueness(self, samples):
        min_num_samples = min(len(samples), self.num_samples_small)
        indices = np.random.choice(len(samples), min_num_samples, replace=False)
        samples_small = [samples[i] for i in indices]
        uniqueness_small = uniqueness(samples_small, fast=self.fast)
        uniqueness_large = uniqueness(samples, fast=self.fast)
        return uniqueness_small, uniqueness_large

    def evaluate_metric(self, metric, dataset, generated):
        fun = METRICS[metric]

        test_set = patch(dataset.get_data("test"))
        generated = patch(generated)

        results = []
        for _ in range(self.num_trials):
            sample = random_sample(generated, n=len(test_set))
            if metric == "nspdk":
                score = mmd.compute_mmd(test_set, sample, metric="nspdk", is_hist=False, n_jobs=48)
                ref_dist, sample_dist = None, None
            else:
                ref, sample = fun(test_set), fun(sample)
                ref_dist, sample_dist = normalize(ref, sample)
                score = entropy(ref_dist + EPS, sample_dist + EPS)
            results.append({
                "model": self.model_name,
                "dataset": dataset.name,
                "metric": metric,
                "score": score,
                "gen": sample_dist,
                "ref": ref_dist
            })

        return results


class Evaluator(EvaluatorBase):
    root = Path("RUNS")


class OrderEvaluator(EvaluatorBase):
    root = Path("RUNS") / "ORDER"
    requires_quantitative = False