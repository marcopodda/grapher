import time
import numpy as np
from pathlib import Path

from .serializer import load_yaml
from .constants import MODEL_NAMES, DATASET_NAMES, ORDER_NAMES, METRIC_NAMES, QUANTITATIVE_METRIC_NAMES

root = Path('RUNS')
order_root = Path('RUNS') / "ORDER"


def to_hms(secs):
    return time.strftime('%H:%M:%S', time.gmtime(secs))


def load_result(model_name, dataset_name, order):
    path = root if order is False else order_root
    path = path / model_name / dataset_name
    result_path = list(path.glob("*"))[-1] / "results" / f"{dataset_name}.yaml"
    return load_yaml(result_path)


def process_result(result, metric_name):
    if metric_name in METRIC_NAMES:
        mean = result[metric_name]["mean"]
        std = result[metric_name]["std"]
        return mean, std
    return result[metric_name], None


def metric_by_dataset(dataset_name, metric_name, order=False):
    scores, stds = [], []
    names = MODEL_NAMES if order is False else ORDER_NAMES
    for model_name in names:
        result = load_result(model_name, dataset_name, order=order)
        score, std = process_result(result, metric_name)
        scores.append(score)
        stds.append(std)

    return scores, stds


def metrics_by_dataset(dataset_name, quantitative=False, order=False):
    metric_names = METRIC_NAMES if quantitative is False else QUANTITATIVE_METRIC_NAMES
    scores_all, stds_all = [], []
    for metric_name in metric_names:
        scores, stds = metric_by_dataset(dataset_name, metric_name, order=order)
        scores_all.append(scores)
        stds_all.append(stds)
    return scores_all, stds_all


def mean_rank(metric_name):
    def get_scores(model_name, metric_name):
        scores = []

        for dataset_name in DATASET_NAMES:
            result = load_result(model_name, dataset_name, order=False)
            score, _ = process_result(result, metric_name)
            scores.append(score)

        return scores

    means_mat = []
    for model_name in MODEL_NAMES:
        means = get_scores(model_name, metric_name)
        means_mat.append(means)
    means_mat = np.array(means_mat)
    return (np.argsort(means_mat.T).mean(axis=0) + 1).tolist()


def mean_ranks(quantitative=False):
    metric_names = METRIC_NAMES if quantitative is False else QUANTITATIVE_METRIC_NAMES
    ranks = []
    for metric_name in metric_names:
        ranks.append(mean_rank(metric_name))
    return ranks
