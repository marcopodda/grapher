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
    models = ["Data", "Ours", "GraphRNN"]
    metrics = [HUMANIZE[m] for m in QUAL_METRICS]
    datasets = [HUMANIZE[d] for d in DATASETS]

    data = collate_results()
    data.to_csv("a.csv")
    data = data[data.Model.isin(models)]

    g = sns.displot(
        x="Value", hue="Model", row="Dataset", col="Metric", kind="kde",
        data=data, common_norm=False, hue_order=models, row_order=datasets, col_order=metrics,
        facet_kws=dict(sharex=False, sharey=False), height=3)

    for i, _ in enumerate(datasets):
        for j, _ in enumerate(metrics):
            g.axes[i,j].set_xlabel("")
            g.axes[i,j].set_ylabel("")

    for i, metric in enumerate(metrics):
        g.axes[-1,i].set_xlabel(metric)

    for i, dataset in enumerate(datasets):
        g.axes[i,0].set_ylabel(dataset)

    g.set_titles(template="", col_template="", row_template="")
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