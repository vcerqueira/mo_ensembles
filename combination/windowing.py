from typing import Union

import numpy as np
import pandas as pd

from common.normalization import normalize_and_proportion


class WindowLoss:

    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def fit(self, Y_hat: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        pass

    def get_weights(self, Y_hat: pd.DataFrame, y: pd.Series):
        se = Y_hat.apply(func=lambda x: (x - y) ** 2, axis=0)
        rolling_mse = se.rolling(window=self.lambda_).mean()

        av_rolling_mse = rolling_mse[self.lambda_:]

        window_weights = av_rolling_mse.apply(func=lambda x: normalize_and_proportion(-x), axis=1)

        out = window_weights.values

        return out
