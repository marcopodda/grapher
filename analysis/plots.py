from matplotlib import pyplot as plt
import seaborn as sns

from analysis.collect import parse_log, collate_scores

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def plot_kde():
    data = collate_scores()
    g = sns.displot(
        x="Value", hue="Model", col="Dataset", row="Metric", kind="kde",
        data=data, common_norm=False,
        facet_kws=dict(sharex=False, sharey=False), height=3)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    plt.savefig("displot.eps")
    plt.clf()


def plot_loss(log_path):
    log = parse_log(log_path)
    hue_order = ["BFS", "DFS", "RANDOM", "BFS RANDOM", "DFS RANDOM", "SMILES"]
    g = sns.FacetGrid(log, col="Dataset", hue="Order", col_wrap=3, hue_order=hue_order, height=3, sharex=False)
    g.map(sns.lineplot, "Epoch", "Loss")
    g.add_legend()
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    plt.savefig("loss.eps")