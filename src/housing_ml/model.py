import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from housing_ml.features import build_preprocess

def build_model() -> TransformedTargetRegressor:
    """ Regressão Linear no log do preço; as previsões volta em escala original """

    pipeline = Pipeline(
        [
            ("preprocess", build_preprocess()),
            ("regressor", LinearRegression())
        ]
    )
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)     
            # TransformedTargetRegressor: transforma o Target (de forma logarítimca) antes do treinamento e reverte essa transformação na hora das previsões