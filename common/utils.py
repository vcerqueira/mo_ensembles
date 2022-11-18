import pickle
import itertools
from typing import Dict

import numpy as np
import pandas as pd


def load_data(filepath):
    with open(filepath, 'rb') as fp:
        data = pickle.load(fp)

    return data


def save_data(data, filepath):
    with open(filepath, 'wb') as fp:
        pickle.dump(data, fp)


def expand_grid(*iters):
    product = list(itertools.product(*iters))
    return {i: [x[i] for x in product]
            for i in range(len(iters))}


def expand_grid_from_dict(x: Dict) -> pd.DataFrame:
    param_grid = expand_grid(*x.values())
    param_grid = pd.DataFrame(param_grid)
    param_grid.columns = x.keys()

    return param_grid


def expand_grid_all(x: Dict) -> Dict:
    param_grid = expand_grid(*x.values())
    new_keys = dict(zip(param_grid.keys(), x.keys()))

    param_grid = {new_keys[k]: v for k, v in param_grid.items()}

    return param_grid


def parse_config(x: pd.Series) -> Dict:
    config = dict(x)

    for key in config:
        try:
            if np.isnan(config[key]):
                config[key] = None
        except TypeError:
            continue

    return config


def key_argmin(x: dict):
    """ Key of the element with maximum value
    :param x: Dictionary
    :return: Key
    """
    x_values = list(x.values())
    x_best_key = list(x.keys())[int(np.argmin(x_values))]

    return x_best_key
