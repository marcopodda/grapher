import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_json(path):
    return json.load(open(path, "r"))


def save_json(obj, path):
    json.dump(obj, open(path, "w"))


def load_pickle(path):
    return pickle.load(open(path, "rb"))


def save_pickle(obj, path):
    pickle.dump(obj, open(path, "wb"))


def load_numpy(path, **kwargs):
    return np.loadtxt(path, **kwargs)


def save_numpy(mat, path, **kwargs):
    np.savetxt(path, mat, **kwargs)


def load_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs)


def save_csv(df, path, **kwargs):
    df.to_csv(path, **kwargs)


def load_yaml(path):
    return yaml.load(open(path, "r"), Loader=yaml.FullLoader)


def save_yaml(obj, path):
    yaml.dump(obj, open(path, "w"))


def load_dict(dict_or_path):
    if isinstance(dict_or_path, dict):
        return dict_or_path

    path = Path(dict_or_path)
    if path.suffix == ".json":
        return load_json(path)
    elif path.suffix in [".yaml", ".yml"]:
        return load_yaml(path)
    elif path.suffix in [".pkl", ".pickle"]:
        return load_pickle(path)

    raise ValueError("Only JSON, YaML and pickle files supported.")


def save_dict(dict_obj, path, serializer, name="config"):
    filename = path / f"{name}.{serializer}"

    if serializer == "json":
        return save_json(dict_obj, filename)
    elif serializer == "yaml":
        return save_yaml(dict_obj, filename)
    elif serializer == "pickle":
        return save_pickle(dict_obj, filename)

    raise ValueError("Only JSON, Yaml and pickle files supported.")
