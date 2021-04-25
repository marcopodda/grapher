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
    # data = data.loc[(data.Dataset==dataset) & (data.Metric==metric),:]
    data = data[data.Model.isin(["Data", "GRAPHRNN"])]
    data = data[data.Dataset.isin(["trees", "ENZYMES"])]
    data = data[data.Metric.isin(["betweenness", "nspdk"])]
    sns.displot(x="Value", row="Dataset", col="Metric", hue="Model", data=data, stat="density", kind="kde", common_norm=False)
    # plt.tight_layout()
    plt.savefig("prova.png")
    plt.clf()