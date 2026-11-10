import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error
)

def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    pred = model.predict(X)

    return {
        "rmse": root_mean_squared_error(y, pred),
        "mae": mean_absolute_error(y, pred),
        "mape": mean_absolute_percentage_error(y, pred),
        "r2": float(r2_score(y, pred))
    }