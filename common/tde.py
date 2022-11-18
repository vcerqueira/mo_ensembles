import pandas as pd


def MultivariateTDE(data: pd.DataFrame, k: int, horizon: int, target_col: str):
    """
    time delay embedding for mv time series

    :param data: multivariate time series as pd.DF
    :param k: embedding dimension (applied to all cols)
    :param horizon: forecasting horizon
    :param target_col: string denoting the target column

    :return: trainable data set
    """

    iter_over_k = list(range(k, 0, -1))

    X_cols = []
    for col in data.columns:
        # input sequence (t-n, ... t-1)
        X, col_iter = [], []
        for i in iter_over_k:
            X.append(data[col].shift(i))

        X = pd.concat(X, axis=1)
        X.columns = [f'{col}-{j}' for j in iter_over_k]
        X_cols.append(X)

    X_cols = pd.concat(X_cols, axis=1)

    # forecast sequence (t, t+1, ... t+n)
    y = []
    for i in range(0, horizon):
        y.append(data[target_col].shift(-i))

    y = pd.concat(y, axis=1)
    y.columns = [f'{target_col}+{i}' for i in range(1, horizon + 1)]

    data_set = pd.concat([X_cols, y], axis=1)

    return data_set


def UnivariateTDE(data: pd.Series, k: int, horizon: int):
    """
    time delay embedding for mv time series

    :param data: multivariate time series as pd.DF
    :param k: embedding dimension (applied to all cols)
    :param horizon: forecasting horizon
    :param target_col: string denoting the target column

    :return: trainable data set
    """

    s = pd.DataFrame({'t': data})
    df = MultivariateTDE(data=s, k=k, horizon=horizon, target_col='t')
    df = df.dropna().reset_index(drop=True)

    return df
