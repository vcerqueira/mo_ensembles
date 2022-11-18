from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from common.normalization import neg_normalize_and_proportion


class Arbitrating:
    """
    ADE
    """

    def __init__(self, lambda_: float):
        """
        :param lambda_: No of recent observations used to trim ensemble
        """
        self.lambda_ = lambda_
        self.meta_model = RandomForestRegressor()

    def fit(self,
            Y_hat_insample: pd.DataFrame,
            y_insample: Union[pd.Series,np.ndarray],
            X_tr: pd.DataFrame):

        Y_hat_insample.reset_index(drop=True, inplace=True)

        if isinstance(y_insample, pd.Series):
            y_insample = y_insample.values

        base_loss = Y_hat_insample.apply(func=lambda x: x - y_insample, axis=0)
        base_loss.reset_index(drop=True, inplace=True)
        print(base_loss)

        self.meta_model.fit(X_tr.reset_index(drop=True), base_loss)

    def get_weights(self, X: pd.DataFrame):
        E_hat = self.meta_model.predict(X)
        E_hat = pd.DataFrame(E_hat).abs()

        W = E_hat.apply(
            func=lambda x: neg_normalize_and_proportion(x),
            axis=1)

        return W.values
