from pathlib import Path

from .serializer import load_yaml
from .constants import MODEL_NAMES, DATASET_NAMES

root = Path('RUNS')
MODEL_NAMES = list(MODEL_NAMES)
MODEL_NAMES.remove("GRAPHER")

def load_results(model_name, dataset_name):
    path = root / model_name / dataset_name
    results_path = list(path.glob("*"))[-1] / "results" / f"{dataset_name}.yaml"
    return load_yaml(results_path)


def process_result(results, metric):
    score = results[metric]["mean"]
    score_std = results[metric]["std"]
    return f"{score:.6f} +/- {score_std:.4f}"


def scores_by_dataset(dataset_name, metric):
    scores = []
    for model_name in MODEL_NAMES:
        results = load_results(model_name, dataset_name)
        score = process_result(results, metric)
        scores.append(f"{model_name:20}: {score}")
    return scores


def scores_by_model(model_name, metric):
    scores = []
    for dataset_name in DATASET_NAMES:
        results = load_results(model_name, dataset_name)
        score = process_result(results, metric)
        scores.append(f"{dataset_name:20}: {score}")
    return scores
