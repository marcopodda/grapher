import time
import torch
import numpy as np
import networkx as nx
from pathlib import Path
from joblib import Parallel, delayed
from functools import partial

from utils import mmd
from utils.serializer import load_yaml
from utils.constants import DATASET_NAMES, MODEL_NAMES, QUALITATIVE_METRIC_NAMES, RUNS_DIR, DATA_DIR
from utils.evaluation import orca, nspdk
from analysis.scoring import load_test_set, score


def score_all():
    SCORES_DIR = Path("SCORES")
    for dataset in DATASET_NAMES:
        test_set = load_test_set(dataset)
        for model in MODEL_NAMES:
            for metric in QUALITATIVE_METRIC_NAMES:
                if not (SCORES_DIR / f"{model}_{dataset}_{metric}.pt").exists():
                    s = score(test_set, model, dataset, metric)
                    torch.save(s, SCORES_DIR / f"{model}_{dataset}_{metric}.pt")

if __name__ == "__main__":
    score_all()