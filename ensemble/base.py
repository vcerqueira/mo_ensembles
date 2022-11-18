from typing import Dict

import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from ensemble.algorithms import METHODS, METHODS_PARAMETERS
from common.utils import expand_grid_all
from common.error import multistep_mae


class MultiOutputEnsemble:

    def __init__(self):
        self.models = {}
        self.validation_error = {}
        self.time = {}
        self.failed = []
        self.selected_methods = []
        self.col_names = []
        self.best_model = ''

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        self.col_names = Y.columns

        for learning_method in METHODS:
            print(f'Creating {learning_method}')
            if len(METHODS_PARAMETERS[learning_method]) > 0:
                gs_df = expand_grid_all(METHODS_PARAMETERS[learning_method])

                n_gs = len(gs_df[[*gs_df][0]])
                for i in range(n_gs):
                    print(f'Training {i} out of {n_gs}')

                    pars = {k: gs_df[k][i] for k in gs_df}
                    pars = {p: pars[p] for p in pars if pars[p] is not None}
                    print(pars)

                    model = METHODS[learning_method](**pars)
                    start = time.time()
                    model.fit(X, Y)
                    end_t = time.time() - start

                    self.models[f'{learning_method}_{i}'] = model
                    self.time[f'{learning_method}_{i}'] = end_t
            else:
                model = METHODS[learning_method]()
                start = time.time()
                model.fit(X, Y)

                end_t = time.time() - start

                self.models[f'{learning_method}_0'] = model
                self.time[f'{learning_method}_0'] = end_t

    def fit_and_trim(self, X, Y, select_percentile: float = .75):

        X_train, X_valid, Y_train, Y_valid = \
            train_test_split(X, Y, test_size=0.1, shuffle=False)

        self.fit(X_train, Y_train)

        Y_hat = self.predict_all(X_valid)

        for m in Y_hat:
            self.validation_error[m] = mean_absolute_error(Y_valid, Y_hat[m])

        err_series = pd.Series(self.validation_error)
        self.best_model = err_series.sort_values().index[0]
        self.selected_methods = err_series[err_series < err_series.quantile(select_percentile)].index.tolist()

        self.fit(X, Y)
        self.models = {k: self.models[k] for k in self.selected_methods}

    def predict_all(self, X: pd.DataFrame):

        preds_all = {}
        for method_ in self.models:
            predictions = self.models[method_].predict(X)
            preds_all[method_] = pd.DataFrame(predictions, columns=self.col_names)

        return preds_all

    @staticmethod
    def get_yhat_by_horizon(y_hat_d: Dict):
        model_names = [*y_hat_d]

        horizon_names = y_hat_d[model_names[0]].columns.tolist()

        yhat_by_horizon_ = {h_: pd.DataFrame({m: y_hat_d[m][h_]
                                              for m in model_names})
                            for h_ in horizon_names}

        return yhat_by_horizon_

    @staticmethod
    def get_yhat_by_model(y_hat_h: Dict):
        fh = [*y_hat_h]

        model_names = y_hat_h[fh[0]].columns.tolist()

        yhat_by_model = {m: pd.DataFrame({h_: y_hat_h[h_][m] for h_ in fh})
                         for m in model_names}

        return yhat_by_model

    def predict(self, X: pd.DataFrame):
        pass

    def combine_predictions(self, Y_hat: Dict[str, pd.DataFrame], Y: pd.DataFrame):

        pass

    def compute_model_loss(self, X: pd.DataFrame, Y: pd.DataFrame):

        """
        this func will have many variations
        :param X:
        :param Y:
        :return:
        """

        model_loss = dict()
        for method in self.models:
            print(f'Computing loss for model: {method}')
            y_hat_model = self.models[method].predict(X)
            y_hat_model = pd.DataFrame(y_hat_model)

            err = pd.DataFrame(y_hat_model.values - Y.values).abs()
            model_loss[method] = err.mean(axis=1)

        model_loss = pd.DataFrame(model_loss)

        return model_loss
