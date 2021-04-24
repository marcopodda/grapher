from pathlib import Path


MODEL_NAMES = ("ER", "BA", "GRU", "GRAPHRNN", "GRAPHER")
DATASET_NAMES = ("ladders", "community", "ego", "trees", "ENZYMES", "PROTEINS_full")
QUALITATIVE_METRIC_NAMES = ("degree", "clustering", "orbit", "betweenness", "nspdk")
QUANTITATIVE_METRIC_NAMES = ("novelty1000", "novelty5000", "uniqueness1000", "uniqueness5000")
ORDER_NAMES = ("random", "bfs-random", "dfs-random", "dfs-fixed", "smiles")

PAD = 0
SOS = 1
EOS = 2

ROOT = Path(".")
RUNS_DIR = ROOT / "RUNS"
ORDER_DIR = RUNS_DIR / "ORDER"
DATA_DIR = ROOT / "DATA"



HUMANIZE_DATASET = {
    "community": "Community",
    "PROTEINS_full": "Protein",
    "ladders": "Ladders",
    "trees": "Trees",
    "ENZYMES": "Enzymes",
    "ego": "Ego"
}

HUMANIZE_ORDER = {
    "random": "Random",
    "bfs-random": "BF Random",
    "dfs-random": "DF Random",
    "dfs-fixed": "DF",
    "smiles": "SMILES",
}

HUMANIZE_METRIC = {
    "degree": "Degree Distribution",
    "clustering": "Clustering Coefficients",
    "orbit": "Orbit Counts",
    "nspdk": "NSPDK"
}
