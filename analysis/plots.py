from matplotlib import pyplot as plt
import seaborn as sns

from utils.constants import HUMANIZE
from analysis.collect import parse_log, collate_results, QUAL_METRICS, DATASETS

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def plot_kde():
    models = ["Data", "GRAPHER", "GRAPHRNN"]
    metrics = [HUMANIZE[m] for m in QUAL_METRICS]
    datasets = [HUMANIZE[d] for d in DATASETS]

    data = collate_results()
    data = data[data.Model.isin(models)]
    data = data[data.Value>0]

    g = sns.displot(
        x="Value", hue="Model", row="Metric", col="Dataset", # kind="kde",
        data=data, common_norm=False, hue_order=models, row_order=metrics, col_order=datasets,
        facet_kws=dict(sharex=False, sharey=False), height=3)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    # g.set_axis_labels(x_var=metrics, y_var=datasets)
    plt.savefig("displot.eps")
    plt.clf()


def plot_loss(log_path):
    log = parse_log(log_path)
    hue_order = ["BFS", "DFS", "RANDOM", "BFS RANDOM", "DFS RANDOM", "SMILES"]
    g = sns.FacetGrid(log, col="Dataset", hue="Order", col_wrap=2, hue_order=hue_order, height=3, sharex=False)
    g.map(sns.lineplot, "Epoch", "Loss")
    g.add_legend()
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    plt.savefig("loss.eps")