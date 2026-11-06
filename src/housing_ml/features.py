""" Esteira de processamento de dados 'numéricos' e 'categóricos'.

- to_frame: obtenção de dados JSON (vindo de APIs), validação e posicionamento das colunas.
obs: algoritmos de ML necessitam estritamente das colunas em posições corretas (são treinados em Matrizes)

"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "price"

NUMERIC_FEATURES = [
    "area_m2", "lot_size_m2", "bedrooms", "bathrooms", "floors", "garage_spots",
    "age_years", "renovation_year", "distance_center_km", "distance_subway_km",
    "crime_index", "school_score", "has_pool", "has_elevator", "latitude", "longitude",
]  # fmt: skip

CATEGORICAL_FEATURES = ["neighborhood", "property_type", "heating", "condition"] 
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def build_preprocess() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler())
        ]
    )

    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES)
        ]
    )

def to_frame(records: list[dict]) -> pd.DataFrame:
    """ Converte registros (ex: payload de API) no DataFrame esperado pelo Pipelie.
    obs: converte dados JSON e estruturas type 'None' para DataFrames e estruturas type NaN (esperado pelo Scikit Learn)"""

    df = pd.DataFrame(records, columns=FEATURES)
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype(float)
    cats = df[CATEGORICAL_FEATURES].astype(object)
    df[CATEGORICAL_FEATURES] = cats.where(cats.notna(), np.nan)

    return df