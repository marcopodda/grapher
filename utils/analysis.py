import time
from pathlib import Path

from .serializer import load_yaml
from .constants import MODEL_NAMES, DATASET_NAMES, ORDER_NAMES

root = Path('RUNS')
order_root = Path('RUNS') / "ORDER"
MODEL_NAMES = ["ER", "BA", "GRU", "GRAPHRNN", "GRAPHER"]
DATASET_NAMES = list(DATASET_NAMES)

def to_hms(secs):
    return time.strftime('%H:%M:%S', time.gmtime(secs))

def load_result(model_name, dataset_name, order):
    path = root if order is False else order_root
    path = path / model_name / dataset_name
    result_path = list(path.glob("*"))[-1] / "results" / f"{dataset_name}.yaml"
    return load_yaml(result_path)


def process_kld(result, metric):
    score = result[metric]["mean"]
    score_std = result[metric]["std"]
    score = "{:.6f} +/- {:.4f}".format(score, score_std)
    return score


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


def klds_by_dataset(dataset_name, metric_name, order=False):
    klds = []
    names = MODEL_NAMES if order is False else ORDER_NAMES
    for model_name in names:
        try:
            result = load_result(model_name, dataset_name, order)
            kld = process_kld(result, metric_name)
            klds.append(kld)
        except:
            if model_name == "smiles":
                klds.append("--")
            else:
                klds.append("0")
    if order is True:
        try:
            model_name = "GRAPHER"
            result = load_result(model_name, dataset_name, False)
            kld = process_kld(result, metric_name)
            klds.append(kld)
        except:
            klds.append("0")

    return " & ".join(klds)


def klds_by_model(model_name, metric_name, order=False):
    klds = []
    for dataset_name in DATASET_NAMES:
        try:
            result = load_result(model_name, dataset_name, order)
            kld = process_kld(result, metric_name)
            klds.append(kld)
        except:
            klds.append("0")

    return r" & ".join(klds)


def metric_by_model(model_name, metric_name, order=False):
    metrics = []
    for dataset_name in DATASET_NAMES:
        try:
            result = load_result(model_name, dataset_name, order)
            metric = process_metric(result, model_name, dataset_name, metric_name)
            metrics.append(metric)
        except:
            metrics.append("0")

    return " & ".join(metrics)


def metric_by_dataset(dataset_name, metric_name, order=False):
    metrics = []
    names = MODEL_NAMES if order is False else ORDER_NAMES
    for model_name in names:
        try:
            result = load_result(model_name, dataset_name, order=order)
            metric = process_metric(result, model_name, dataset_name, metric_name)
            metrics.append(metric)
        except:
            metrics.append("0")

    string = " & ".join(metrics)
    string.replace(r"\\", r"\\")
    return string