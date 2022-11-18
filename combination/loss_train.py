from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse

from common.normalization import normalize_and_proportion


class LossTrain:

    def __init__(self):
        self.rmse = None
        self.weights = None

    def fit(self, Y_hat: pd.DataFrame, y: Union[pd.Series,np.ndarray]):
        self.rmse = pd.Series({k: mse(y, Y_hat[k], squared=False)
                               for k in Y_hat})

        self.weights = normalize_and_proportion(-self.rmse)

    def get_weights(self, Y_hat: pd.DataFrame):
        weights = np.array([self.weights, ] * Y_hat.shape[0])

        return weights
