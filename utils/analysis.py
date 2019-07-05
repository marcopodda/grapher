from pathlib import Path

from .serializer import load_yaml
from .constants import MODEL_NAMES, DATASET_NAMES

root = Path('RUNS')
MODEL_NAMES = list(MODEL_NAMES)
MODEL_NAMES.remove("GRAPHER")

def load_result(model_name, dataset_name):
    path = root / model_name / dataset_name
    result_path = list(path.glob("*"))[-1] / "results" / f"{dataset_name}.yaml"
    return load_yaml(result_path)


def process_kld(result, metric):
    score = result[metric]["mean"]
    score_std = result[metric]["std"]
    return f"{score:.6f} +/- {score_std:.4f}"


def process_metric(result, metric_name):
    metric1000 = result[f'{metric_name}1000']
    metric10000 = result[f'{metric_name}10000']
    return f"{metric_name.capitalize()}@1000 samples: {metric1000:.6f}" + \
           f"{metric_name.capitalize()}@10000 samples: {metric10000:.6f}"


def klds_by_dataset(dataset_name, metric):
    klds = []
    for model_name in MODEL_NAMES:
        result = load_result(model_name, dataset_name)
        kld = process_kld(result, metric)
        klds.append(f"{model_name:20}: {kld}")
    return klds


def klds_by_model(model_name, metric):
    klds = []
    for dataset_name in DATASET_NAMES:
        result = load_result(model_name, dataset_name)
        kld = process_kld(result, metric)
        klds.append(f"{dataset_name:20}: {kld}")
    return klds


def metric_by_model(model_name, metric_name):
    metrics = []
    for dataset_name in DATASET_NAMES:
        result = load_result(model_name, dataset_name)
        metric = process_metric(result, metric_name)
        metrics.append(metric)
    return metrics


def metric_by_dataset(dataset_name, metric_name):
    metrics = []
    for model_name in MODEL_NAMES:
        result = load_result(model_name, dataset_name)
        metric = process_metric(result, metric_name)
        metrics.append(metric)
    return metrics