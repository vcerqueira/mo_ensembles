import warnings

from common.error import multistep_mae
from ensemble.base import MultiOutputEnsemble
from weighting.weights_over_h import Weighting

warnings.simplefilter('ignore', UserWarning)


def cval_cycle(X_tr, Y_tr, X_ts, Y_ts):
    """
    :param X_tr:
    :param Y_tr:
    :param X_ts:
    :param Y_ts:
    :return:
    """

    base = MultiOutputEnsemble()
    base.fit_and_trim(X_tr, Y_tr)

    Y_hat_tr = base.predict_all(X_tr)
    Y_hat_tr_h = base.get_yhat_by_horizon(Y_hat_tr)
    Y_hat = base.predict_all(X_ts)
    Y_hat_h = base.get_yhat_by_horizon(Y_hat)

    weights = Weighting(Y_hat_h=Y_hat_h,
                        Y_hat_h_insample=Y_hat_tr_h,
                        Y=Y_ts,
                        Y_insample=Y_tr,
                        X=X_ts,
                        X_insample=X_tr,
                        lambda_=50,
                        omega=0.5)

    Y_hat_f = weights.propagate_from_t1()
    Y_hat_f2 = weights.complete_fh()
    Y_hat_f3 = weights.direct_fh()
    Y_hat_f4 = weights.propagate_from_lt()

    yhf = {**Y_hat_f, **Y_hat_f2, **Y_hat_f3, **Y_hat_f4}

    err = multistep_mae(yhf, Y_ts)

    return err
