from sklearn.ensemble \
    import (RandomForestRegressor,
            ExtraTreesRegressor,
            BaggingRegressor)
from sklearn.linear_model \
    import (Lasso,
            Ridge,
            OrthogonalMatchingPursuit,
            ElasticNet)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression, PLSCanonical

from common.utils import expand_grid_from_dict

METHODS = \
    dict(
        RandomForestRegressor=RandomForestRegressor,
        PLSRegression=PLSRegression,
        PLSCanonical=PLSCanonical,
        ExtraTreesRegressor=ExtraTreesRegressor,
        OrthogonalMatchingPursuit=OrthogonalMatchingPursuit,
        Lasso=Lasso,
        KNeighborsRegressor=KNeighborsRegressor,
        Ridge=Ridge,
        ElasticNet=ElasticNet,
        BaggingRegressor=BaggingRegressor,
    )

METHODS_PARAMETERS = \
    dict(
        RandomForestRegressor={
            'n_estimators': [50, 100],
            'max_depth': [None, 3, 5],
        },
        ExtraTreesRegressor={
            'n_estimators': [50, 100],
            'max_depth': [None, 3, 5],
        },
        OrthogonalMatchingPursuit={},
        Lasso={
            'alpha': [1, .5, .25, .75]
        },
        KNeighborsRegressor={
            'n_neighbors': [1, 5, 10, 20, 50],
            'weights': ['uniform', 'distance'],
        },
        Ridge={
            'alpha': [1, .5, .25, .75]
        },
        ElasticNet={
        },
        PLSRegression={
            'n_components': [2, 3, 5]
        },
        PLSCanonical={
            'n_components': [2, 3, 5]
        },
        BaggingRegressor={
            'n_estimators': [50, 100]
        },
    )

n_models = 0
for k in METHODS_PARAMETERS:
    if len(METHODS_PARAMETERS[k]) > 0:
        n_models += expand_grid_from_dict(METHODS_PARAMETERS[k]).shape[0]
    else:
        n_models += 1

print(n_models)
