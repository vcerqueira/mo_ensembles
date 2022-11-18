from typing import Dict

import pandas as pd
from sklearn.metrics import mean_absolute_error


def multistep_mae(predictions: Dict, Y_ts: pd.DataFrame):
    """
    Computing MAE for multiple horizons
    """
    err = {}
    for m in predictions:
        err_k = {col: mean_absolute_error(
            Y_ts[col].values,
            predictions[m][col].values) for col in Y_ts.columns}

        err[m] = err_k

    err_df = pd.DataFrame(err)

    return err_df
