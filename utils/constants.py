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


HUMANIZE = {
    "PROTEINS_full": "PROTEINS",
    "trees": "TREES",
    "ladders": "LADDERS",
    "community": "COMMUNITY",
    "ego": "EGO",
    "ENZYMES": "ENZYMES",
    "bfs-fixed": "BFS",
    "dfs-fixed": "DFS",
    "random": "RANDOM",
    "dfs-random": "DFS RANDOM",
    "bfs-random": "BFS RANDOM",
    "smiles": "SMILES",
    "degree": "Degree Dist.",
    "clustering": "Clustering Coef.",
    "orbit": "Orbit Counts",
    "nspdk": "NSPDK",
    "betweenness": "Beetweenness C.",
    "novelty1000": "Novelty@1000",
    "novelty5000": "Novelty@5000",
    "uniqueness1000": "Uniqueness@1000",
    "uniqueness5000": "Uniqueness@5000",
    "time1000": "Time@1000",
    "time5000": "Time@5000"
}