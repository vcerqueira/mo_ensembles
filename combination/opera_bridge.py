from typing import Union

import numpy as np
import pandas as pd
from rpy2.robjects import pandas2ri
import rpy2.robjects as r_objects

r_objects.r(
    """
    OPERA_r_port <-
      function(Y_hat,Y, model) {
        library(opera)

        MIXTURE <- mixture(model = model, loss.type = "square")
        for (i in 1:length(Y)) {
          suppressWarnings(MIXTURE <- predict(MIXTURE, newexperts = Y_hat[i, ], newY = Y[i]))
        }

        W <- MIXTURE$weights
        
        return(W)
      }
    """
)

opera_function = r_objects.globalenv['OPERA_r_port']


class OperaR:

    def __init__(self, method: str):
        assert method in ['Ridge', 'MLpol', 'FS', 'EWA']

        self.method = method
        self.weights = None

    def fit(self, Y_hat: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        pass

    def get_weights(self,
                    Y_hat: pd.DataFrame,
                    y: pd.Series):
        pandas2ri.activate()

        y_ = pandas2ri.py2rpy_pandasseries(pd.Series(y))
        Y_hat_ = pandas2ri.py2rpy_pandasdataframe(Y_hat)

        self.weights = opera_function(Y_hat_, y_, self.method)

        pandas2ri.deactivate()

        return self.weights
