import itertools
import time
from joblib.parallel import Parallel, delayed
import torch
import numpy as np
import networkx as nx
from pathlib import Path
from functools import partial

from dataset import load_dataset
from .experiment import load_experiment
from .eval import (
    degree_dist,
    clustering_dist,
    orbit_dist,
    betweenness_dist,
    nspdk_dist,
    patch,
    random_sample,
    novelty,
    uniqueness)

from utils import mmd
from utils.constants import DATASET_NAMES


METRICS = {
    "degree": {
        "fun": degree_dist,
        "mmd_kwargs": dict(metric=mmd.gaussian_emd, is_hist=True, n_jobs=40)
    },
    "clustering": {
        "fun": clustering_dist,
        "mmd_kwargs": dict(metric=partial(mmd.gaussian_emd, sigma=0.1, distance_scaling=100), is_hist=True, n_jobs=40)
    },
    "orbit": {
        "fun": orbit_dist,
        "mmd_kwargs": dict(metric=partial(mmd.gaussian_emd, sigma=30.0), is_hist=True, n_jobs=40)
    },
    "betweenness": {
        "fun": betweenness_dist,
        "mmd_kwargs": dict(metric=mmd.gaussian_emd, is_hist=True, n_jobs=40)
    },
    "nspdk": {
        "fun": nspdk_dist,
        "mmd_kwargs": dict(metric="nspdk", is_hist=False, n_jobs=40)
    },
}


class EvaluatorBase:
    requires_quantitative = True
    def __init__(self, model_name):
        self.model_name = model_name
        self.num_samples = 5000
        self.num_samples_small = 1000
        self.num_samples_metric = 100
        self.num_trials = 3

    def novelty_not_calculated(self, result):
        return result.novelty_not_calculated

    def uniqueness_not_calculated(self, result):
        return result.uniqueness_not_calculated

    def evaluate(self):
        for dataset_name in DATASET_NAMES:
            if self.model_name == "smiles" and dataset_name not in ["PROTEINS_full", "ENZYMES"]:
                continue
            print(f"Evaluating {dataset_name}...")
            exp = load_experiment(self.root, self.model_name, dataset_name)
            dataset = load_dataset(dataset_name, self.model_name, exp)

            path = exp.root / "results" / f"results.pt"
            if not path.exists():
                result = {}
                samples = self.get_samples(exp)

                if self.requires_quantitative:
                    print("\tCalculating novelty...")
                    novelty_small, novelty_large = self.evaluate_novelty(dataset, samples)
                    print("\tCalculating uniqueness...")
                    uniqueness_small, uniqueness_large = self.evaluate_uniqueness(samples)
                    result.update(**{
                        f"novelty{self.num_samples}": novelty_large,
                        f"uniqueness{self.num_samples}": uniqueness_large,
                        f"novelty{self.num_samples_small}": novelty_small,
                        f"uniqueness{self.num_samples_small}": uniqueness_small
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
                print("\tAlready evaluated, skipping.")

    def get_samples(self, exp):
        time_elapsed = None
        filename = f"samples.pt"

        if not (exp.root / "samples" / filename).exists():
            print("\tGetting samples...", end=" ")
            start = time.time()
            P = Parallel(n_jobs=32, verbose=0)
            samples = P(delayed(exp.sample)(1) for _ in range(self.num_samples))
            # samples = exp.sample(num_samples=self.num_samples)
            time_elapsed = time.time() - start
            with open(exp.root / "samples" / "elapsed.txt", "w") as f:
                print(time_elapsed, file=f)
            samples = list(itertools.chain.from_iterable(samples))
            torch.save(samples, exp.root / "samples" / filename)
            print("\tDone.")
        else:
            print("\tSamples ready.")

        samples = torch.load(exp.root / "samples" / filename)
        return [G for G in samples if G.number_of_nodes() > 1 and G.number_of_edges() > 0]

    def evaluate_novelty(self, dataset, samples):
        train_data = dataset.get_data('train')
        min_num_samples = min(len(samples), self.num_samples_small)
        indices = np.random.choice(len(samples), min_num_samples, replace=False)
        samples_small = [samples[i] for i in indices]
        novelty_small = novelty(train_data, samples_small)
        novelty_large = novelty(train_data, samples)
        return novelty_small, novelty_large

    def evaluate_uniqueness(self, samples):
        min_num_samples = min(len(samples), self.num_samples_small)
        indices = np.random.choice(len(samples), min_num_samples, replace=False)
        samples_small = [samples[i] for i in indices]
        uniqueness_small = uniqueness(samples_small)
        uniqueness_large = uniqueness(samples)
        return uniqueness_small, uniqueness_large

    def evaluate_metric(self, metric, dataset, samples):
        fun = METRICS[metric]["fun"]
        mmd_kwargs = METRICS[metric]["mmd_kwargs"]

        test_set = patch(dataset.get_data("test"))
        samples = patch(samples)

        num_samples = min(len(test_set), self.num_samples_metric)

        results = []
        for _ in range(self.num_trials):
            gen = random_sample(samples, n=num_samples)
            ref = random_sample(test_set, n=num_samples)

            gen_dist = fun(gen)
            test_dist = fun(ref)

            score = mmd.compute_mmd(test_dist, gen_dist, **mmd_kwargs)
            results.append({
                "model": self.model_name,
                "dataset": dataset.name,
                "metric": metric,
                "score": score,
                "gen": gen_dist,
                "ref": test_dist
            })

        return results


class Evaluator(EvaluatorBase):
    root = Path("RUNS")


class OrderEvaluator(EvaluatorBase):
    root = Path("RUNS") / "ORDER"
    requires_quantitative = False