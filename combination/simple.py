from typing import Union

import numpy as np
import pandas as pd


class Simple:

    def __init__(self):
        pass

    def fit(self, Y_hat: pd.DataFrame, y: Union[pd.Series,np.ndarray]):
        pass

    @staticmethod
    def get_weights(Y_hat: pd.DataFrame):
        eq_weights = np.ones_like(Y_hat) / Y_hat.shape[1]

        return eq_weights
