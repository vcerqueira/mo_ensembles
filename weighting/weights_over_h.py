from typing import Dict

import numpy as np
import pandas as pd

from combination.ade import Arbitrating
from combination.best import BestSingle
from combination.loss_train import LossTrain
from combination.opera_bridge import OperaR
from combination.simple import Simple
from combination.windowing import WindowLoss
from committees.external import ExternalCommittee
from committees.internal import Committee
from ensemble.base import MultiOutputEnsemble


class Weighting:

    def __init__(self,
                 Y_hat_h: Dict[str, pd.DataFrame],
                 Y_hat_h_insample: Dict[str, pd.DataFrame],
                 Y: pd.DataFrame,
                 Y_insample: pd.DataFrame,
                 X: pd.DataFrame,
                 X_insample: pd.DataFrame,
                 lambda_: int,
                 omega: float):
        """

        :param Y_hat_h: Dict with preds k:v-> horizon:DF(n_obs,n_models)
        :param Y_hat_h_insample:
        :param Y:
        :param Y_insample:
        :param X:
        :param X_insample:
        :param lambda_:
        :param omega:
        """
        self.Y_hat_h = Y_hat_h
        self.horizon = [*self.Y_hat_h]
        self.Y_hat_h_insample = Y_hat_h_insample
        self.Y = Y
        self.Y_insample = Y_insample
        self.X = X
        self.X_insample = X_insample
        self.lambda_ = lambda_
        self.omega = omega
        self.weights = None

    def propagate_from_t1(self):

        weights_at_t1 = \
            self.weights_by_method(Y_hat=self.Y_hat_h['t+1'],
                                   Y_hat_insample=self.Y_hat_h_insample['t+1'],
                                   y=self.Y['t+1'],
                                   y_insample=self.Y_insample['t+1'],
                                   X=self.X,
                                   X_insample=self.X_insample,
                                   lambda_=self.lambda_,
                                   omega=0.5,
                                   burn_in=1)

        self.weights = weights_at_t1

        print('Combining')
        y_hat_final = {}
        for method in weights_at_t1:
            h_out = {}
            for h_ in self.horizon:
                yh = self.Y_hat_h[h_]
                w = weights_at_t1[method]

                assert w.shape == yh.shape, 'shapes do not conform'

                h_out[h_] = self.predict_on_batch(Y_hat=yh, weights=w)

            y_hat_final[f'{method}_T1Forw'] = pd.DataFrame(h_out)

        return y_hat_final

    def propagate_from_lt(self):
        fh = self.horizon[-1]

        weights_at_lt = \
            self.weights_by_method(Y_hat=self.Y_hat_h[fh],
                                   Y_hat_insample=self.Y_hat_h_insample[fh],
                                   y=self.Y[fh],
                                   y_insample=self.Y_insample[fh],
                                   X=self.X,
                                   X_insample=self.X_insample,
                                   lambda_=self.lambda_,
                                   omega=0.5,
                                   burn_in=18)

        self.weights = weights_at_lt

        print('Combining')
        y_hat_final = {}
        for method in weights_at_lt:
            h_out = {}
            for h_ in self.horizon:
                yh = self.Y_hat_h[h_]
                w = weights_at_lt[method]

                assert w.shape == yh.shape, 'shapes do not conform'

                h_out[h_] = self.predict_on_batch(Y_hat=yh, weights=w)

            y_hat_final[f'{method}_LTBack'] = pd.DataFrame(h_out)

        return y_hat_final

    def complete_fh(self):

        Y_hat_m = MultiOutputEnsemble.get_yhat_by_model(self.Y_hat_h)
        Y_hat_m_insample = MultiOutputEnsemble.get_yhat_by_model(self.Y_hat_h_insample)

        Y_hat_aux, Y_hat_aux_insample = {}, {}
        for m in Y_hat_m:
            yh = Y_hat_m[m].reset_index(drop=True)
            yh_ins = Y_hat_m_insample[m].reset_index(drop=True)

            assert yh.shape == self.Y.shape
            assert yh_ins.shape == self.Y_insample.shape

            self.Y.reset_index(drop=True, inplace=True)
            self.Y_insample.reset_index(drop=True, inplace=True)

            Y_hat_aux[m] = (yh - self.Y).abs().mean(axis=1)
            Y_hat_aux_insample[m] = (yh_ins - self.Y_insample).abs().mean(axis=1)

        Y_hat_aux = pd.DataFrame(Y_hat_aux)
        Y_hat_aux_insample = pd.DataFrame(Y_hat_aux_insample)

        y_aux = np.zeros(Y_hat_aux.shape[0])
        y_ins_aux = np.zeros(Y_hat_aux_insample.shape[0])

        weights_at_compl = \
            self.weights_by_method(Y_hat=Y_hat_aux,
                                   Y_hat_insample=Y_hat_aux_insample,
                                   y=y_aux,
                                   y_insample=y_ins_aux,
                                   X=self.X,
                                   X_insample=self.X_insample,
                                   lambda_=self.lambda_,
                                   omega=0.5,
                                   burn_in=18)

        print('Combining')
        y_hat_final = {}
        for method in weights_at_compl:
            h_out = {}
            for h_ in self.horizon:
                yh = self.Y_hat_h[h_]
                w = weights_at_compl[method]

                assert w.shape == yh.shape, 'shapes do not conform'

                h_out[h_] = self.predict_on_batch(Y_hat=yh, weights=w)

            y_hat_final[f'{method}_Comp'] = pd.DataFrame(h_out)

        return y_hat_final

    def direct_fh(self):

        weights_by_fh = {}
        for i, fh in enumerate(self.horizon):
            weights_by_fh[fh] = \
                self.weights_by_method(Y_hat=self.Y_hat_h[fh],
                                       Y_hat_insample=self.Y_hat_h_insample[fh],
                                       y=self.Y[fh],
                                       y_insample=self.Y_insample[fh],
                                       X=self.X,
                                       X_insample=self.X_insample,
                                       lambda_=self.lambda_,
                                       omega=0.5,
                                       burn_in=i + 1)

        print('Combining')
        y_hat_final = {}
        for method in weights_by_fh[self.horizon[0]]:
            h_out = {}
            for h_ in self.horizon:
                yh = self.Y_hat_h[h_]
                w = weights_by_fh[h_][method]

                assert w.shape == yh.shape, 'shapes do not conform'

                h_out[h_] = self.predict_on_batch(Y_hat=yh, weights=w)

            y_hat_final[f'{method}_Dir'] = pd.DataFrame(h_out)

        return y_hat_final

    @staticmethod
    def predict_on_batch(Y_hat: pd.DataFrame, weights: pd.DataFrame):

        if isinstance(Y_hat, pd.DataFrame):
            Y_hat = Y_hat.values

        if isinstance(weights, pd.DataFrame):
            weights = weights.values

        n = Y_hat.shape[0]

        pred = np.zeros(n)
        for i in range(n):
            pred[i] = np.sum(Y_hat[i] * weights[i])

        return pred

    @staticmethod
    def weights_by_method(Y_hat: pd.DataFrame,
                          Y_hat_insample: pd.DataFrame,
                          y: np.ndarray,
                          y_insample: np.ndarray,
                          X: pd.DataFrame,
                          X_insample: pd.DataFrame,
                          lambda_: int,
                          omega: float,
                          burn_in: int = None):

        Y_hat_comb = pd.concat([Y_hat_insample, Y_hat], ignore_index=True).reset_index(drop=True)
        y_comb = np.concatenate([y_insample, y])

        agg_methods = {
            'ADE': Arbitrating(lambda_=lambda_),
            'Best': BestSingle(),
            'LossTrain': LossTrain(),
            'MLpol': OperaR(method='MLpol'),
            # 'Ridge': OperaR(method='Ridge'),
            'EWA': OperaR(method='EWA'),
            'FS': OperaR(method='FS'),
            'Simple': Simple(),
            'Window': WindowLoss(lambda_=lambda_),
            # 'Oracle': Oracle(),
            'Oracle': LossTrain(),
        }

        COMMITTEE_LIST = ['Window', 'FS', 'EWA', 'MLpol']

        print('Fitting aggregation functions')
        for k in agg_methods:
            if k == 'ADE':
                agg_methods[k].fit(Y_hat_insample, y_insample, X_insample)
            elif k == 'Oracle':
                agg_methods[k].fit(Y_hat, y)
            else:
                agg_methods[k].fit(Y_hat_insample, y_insample)

        print('Getting weights')
        weights = {}
        for k in agg_methods:
            if k == 'ADE':
                weights[k] = agg_methods[k].get_weights(X)
            elif k in ['MLpol', 'Window', 'Ridge', 'EWA', 'FS']:
                weights[k] = agg_methods[k].get_weights(Y_hat_comb, y_comb)
            else:
                weights[k] = agg_methods[k].get_weights(Y_hat_comb)

        print('Subset weights')
        for k in weights:
            if burn_in is not None and k != 'ADE':
                weights[k] = weights[k][:-(burn_in + 1)]

            weights[k] = weights[k][-Y_hat.shape[0]:]

        print('Blast')
        blast = Committee(omega=0, weights=weights['Window'], n_models=1)
        blast_weights = blast.from_weights(weights['Window'].copy())
        weights['Blast'] = blast_weights

        for m in COMMITTEE_LIST:
            comm_m = Committee(omega=omega, weights=weights[m], n_models=1)
            weights[f'COMM_{m}'] = comm_m.from_weights(weights[m].copy())

        ext_comm = ExternalCommittee(omega=omega,
                                     weights=weights['ADE'],
                                     col_names=Y_hat_insample.columns.to_list())

        ade_c = ext_comm.from_weights(target_weights=weights['ADE'],
                                      source_weights=weights['Window'])

        weights['ADE'] = ade_c

        return weights
