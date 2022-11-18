from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse


class BestSingle:

    def __init__(self):
        self.best_model = None

    def fit(self, Y_hat: pd.DataFrame, y: Union[pd.Series,np.ndarray]):
        rmse = pd.Series({k: mse(y, Y_hat[k], squared=False)
                          for k in Y_hat})

        self.best_model = pd.Series(rmse).sort_values().index[0]

    def get_weights(self, Y_hat):
        assert self.best_model in Y_hat.columns, 'best model not in preds'

        weights = np.zeros_like(Y_hat)
        weights = pd.DataFrame(weights, columns=Y_hat.columns)
        weights[self.best_model] = 1

        return weights.values
