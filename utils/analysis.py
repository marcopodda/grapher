import time
from pathlib import Path

from .serializer import load_yaml
from .constants import MODEL_NAMES, DATASET_NAMES

root = Path('RUNS')
MODEL_NAMES = ["GRAPHRNN", "GRAPHER", "GRU", "BA", "ER"]
# DATASET_NAMES = ["ENZYMES", "PROTEINS_full", "community"]

def to_hms(secs):
    return time.strftime('%H:%M:%S', time.gmtime(secs))

def load_result(model_name, dataset_name):
    path = root / model_name / dataset_name
    result_path = list(path.glob("*"))[-1] / "results" / f"{dataset_name}.yaml"
    return load_yaml(result_path)


def process_kld(result, metric):
    score = result[metric]["mean"]
    score_std = result[metric]["std"]
    return f"{score:.6f}\pm{score_std:.4f}"


def process_metric(result, model_name, dataset_name, metric_name):
    names = sorted([name for name in result if metric_name in name])

    metric1 = result[names[0]]
    metric2 = result[names[1]]

    if metric_name == 'time':
        metric1 = to_hms(metric1)
        metric2 = to_hms(metric2)
        return f"{model_name:20} - {dataset_name:20} - " + \
           f"{names[0].capitalize()} samples: {metric1} " + \
           f"{names[1].capitalize()} samples: {metric2}"

    return f"{model_name:20} - {dataset_name:20} - " + \
           f"{names[0].capitalize()} samples: {float(metric1):.6f} " + \
           f"{names[1].capitalize()} samples: {float(metric2):.6f}"


def klds_by_dataset(dataset_name, metric_name):
    klds = []
    for model_name in MODEL_NAMES:
        result = load_result(model_name, dataset_name)
        kld = process_kld(result, metric_name)
        klds.append(f"{model_name:20}: {kld}")
    return klds


def klds_by_model(model_name, metric_name):
    klds = []
    for dataset_name in DATASET_NAMES:
        result = load_result(model_name, dataset_name)
        kld = process_kld(result, metric_name)
        klds.append(f"{dataset_name:20}: {kld}")
    return klds


def metric_by_model(model_name, metric_name):
    metrics = []
    for dataset_name in DATASET_NAMES:
        result = load_result(model_name, dataset_name)
        metric = process_metric(result, model_name, dataset_name, metric_name)
        metrics.append(metric)
    return metrics


def metric_by_dataset(dataset_name, metric_name):
    metrics = []
    for model_name in MODEL_NAMES:
        result = load_result(model_name, dataset_name)
        metric = process_metric(result, model_name, dataset_name, metric_name)
        metrics.append(metric)
    return metrics