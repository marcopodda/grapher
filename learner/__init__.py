from .experiment import (
    Experiment, GraphRNNExperiment, OrderExperiment,
    ERDegreeExperiment, ERClusteringExperiment, BADegreeExperiment, BAClusteringExperiment, GRUExperiment)


def get_exp_class(model_name):
    if model_name == "GRAPHER":
        return Experiment

    if model_name == "GRAPHRNN":
        return GraphRNNExperiment

    if model_name == "ER_degree":
        return ERDegreeExperiment

    if model_name == "ER_clustering":
        return ERClusteringExperiment

    if model_name == "BA_degree":
        return BADegreeExperiment

    if model_name == "BA_clustering":
        return BAClusteringExperiment

    if model_name == "GRU":
        return GRUExperiment

    return OrderExperiment
