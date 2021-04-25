from matplotlib import pyplot as plt
import seaborn as sns

from analysis.analyze import collate_experiments, collate_order_experiments

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def plot_metric_kde():
    _, data = collate_experiments()
    data = data[data.Metric.isin(["degree", "clustering", "orbits"])]
    data = data[data.Dataset!="PROTEINS_full"]
    data = data[data.Model.isin(["Data", "GRAPHER", "GRAPHRNN"])]
    facet_kws = {"sharex": False, "sharey": False}
    sns.displot(x="Value", row="Dataset", col="Metric", hue="Model", data=data, kind="kde", facet_kws=facet_kws, common_norm=False)
    plt.savefig("distplot.eps")
    plt.clf()


def plot_m(dataset, metric):
    _, data = collate_experiments()
    data = data[data.Dataset==dataset]
    data = data[data.Metric==metric]
    data = data[data.Model.isin(["Data", "GRAPHER", "GRAPHRNN"])]
    sns.distplot(data[data.Model=="Data"].Value, hist=False, label="Data")
    sns.distplot(data[data.Model=="GRAPHER"].Value, hist=False, label="GRAPHER")
    sns.distplot(data[data.Model=="GRAPHRNN"].Value, hist=False, label="GRAPHRNN")

    plt.tight_layout()
    plt.savefig(f"{dataset}-{metric}.eps")
    plt.clf()