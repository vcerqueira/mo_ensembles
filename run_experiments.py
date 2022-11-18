import os
import warnings

import pandas as pd
from numpy.linalg import LinAlgError
from gluonts.dataset.repository.datasets import get_dataset, dataset_names
from rpy2.rinterface_lib.embedded import RRuntimeError

from workflows.cross_val_outer import cross_val_workflow

warnings.simplefilter("ignore", UserWarning)

ALL_DATASETS = [
    'nn5_daily_without_missing',
    'solar-energy',
    'traffic_nips',
    'electricity_nips',
    'taxi_30min',
    'm4_hourly',
    'm4_weekly'
]

K = 5
H = 18

DS = 'm4_weekly'

dataset = get_dataset(DS, regenerate=False)

train = list(dataset.train)
train = [x['target'] for x in train]

for i, ds in enumerate(train):
    print(i)
    # ds = train[0]
    file_name = f'{DS}_ts_{i}.csv'
    if file_name in os.listdir('results'):
        continue
    else:
        pd.DataFrame().to_csv(f'results/{file_name}')

    series = pd.Series(ds)
    if len(ds) < 500:
        continue

    try:
        series_result = cross_val_workflow(series, k=K, h=H)
    except (ValueError, LinAlgError, IndexError, RRuntimeError) as e:
        continue

    if len(series_result) == 0:
        continue

    err_df = pd.concat(series_result)
    err_avg = err_df.groupby(err_df.index).mean()

    err_avg.to_csv(f'results/{file_name}')
