import pandas as pd

from common.monte_carlo import MonteCarloCV
from common.tde import UnivariateTDE
from workflows.cross_val_inner import cval_cycle

CV_SPLIT_TRAIN_SIZE, CV_SPLIT_TEST_SIZE = 0.6, 0.1
CV_N_SPLITS = 1
APPLY_DIFF = True


def cross_val_workflow(series, k, h):
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    if APPLY_DIFF:
        series = series.diff()

    df = UnivariateTDE(series, k=k, horizon=h)

    is_future = df.columns.str.contains('\+')

    X = df.iloc[:, ~is_future]
    Y = df.iloc[:, is_future]

    mc = MonteCarloCV(n_splits=CV_N_SPLITS,
                      train_size=CV_SPLIT_TRAIN_SIZE,
                      test_size=CV_SPLIT_TEST_SIZE,
                      gap=h + k)

    err_list = []
    for tr_idx, ts_idx in mc.split(X, Y):
        X_tr = X.iloc[tr_idx, :]
        Y_tr = Y.iloc[tr_idx, :]
        X_ts = X.iloc[ts_idx, :]
        Y_ts = Y.iloc[ts_idx, :]
        X_tr_ = X_tr.head(-k - h)
        Y_tr_ = Y_tr.head(-k - h)

        print('Running inner pipeline')
        err = cval_cycle(X_tr_, Y_tr_, X_ts, Y_ts)
        err_list.append(err)

    return err_list

