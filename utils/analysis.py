import numpy as np

from .constants import *
from .misc import load_result
from .serializer import load_yaml


def process_result(result, metric_name):
    if metric_name in QUALITATIVE_METRIC_NAMES:
        mean = result[metric_name]["mean"]
        std = result[metric_name]["std"]
        return mean, std
    return result[metric_name], None


def metric_by_dataset(dataset_name, metric_name, order=False):
    scores, stds = [], []
    model_names = MODEL_NAMES if order is False else ORDER_NAMES
    for model_name in model_names:
        try:
            result = load_result(model_name, dataset_name, order=order)
            score, std = process_result(result, metric_name)
            scores.append({f"{dataset_name}-{metric_name}-mean": score})
            stds.append({f"{dataset_name}-{metric_name}-std": std})
        except:
            continue

    return scores, stds


def metrics_by_dataset(dataset_name, quantitative=False, order=False):
    if quantitative:
        metric_names = QUANTITATIVE_METRIC_NAMES
    else:
        metric_names = QUALITATIVE_METRIC_NAMES

    scores_all, stds_all = [], []
    for metric_name in metric_names:

        scores, stds = metric_by_dataset(dataset_name, metric_name, order=order)
        if scores != []:
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
    if quantitative:
        metric_names = QUANTITATIVE_METRIC_NAMES
    else:
        metric_names = QUALITATIVE_METRIC_NAMES

    ranks = []
    for metric_name in metric_names:
        ranks.append(mean_rank(metric_name))
    return ranks
