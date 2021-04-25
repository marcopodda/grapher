from analysis.scoring import load_test_set, score_all


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