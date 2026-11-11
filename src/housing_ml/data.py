""" Validar os dados antes de processar. 
Regras de filtragem e validação da qualidade dos dados 
"""

from housing_ml.logging_config import setup_logger
from pathlib import Path
import pandas as pd
from housing_ml.features import FEATURES, TARGET

logger = setup_logger(level="INFO", fmt="text", log_file=Path("../../artifacts/logs/data.log"))

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """ Remove linhas sem alvo ou com valores fisicamente impossíveis """

    invalid = {
        "price nulo ou <= 0": ~df[TARGET].gt(0),
        "area_m2 fora de [0, 1000]": ~df["area_m2"].between(10, 1000),
        "age_years fora de [0, 200]": ~df["age_years"].between(0, 200)
    }

    for reason, mask in invalid.items():
        if mask.any():
            logger.warning("%d linhas com %s", mask.sum(), reason)

    to_drop = pd.concat(invalid.values(), axis=1).any(axis=1)
    logger.info("Limpeza: %d -> %d linhas (%d removidas)", len(df), (~to_drop).sum(), to_drop.sum())

    return df.loc[~to_drop].reset_index(drop=True)  # ~to_drop: retorna as linhas válidas

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists:
        raise FileNotFoundError(f"Dataset não encontrado em '{path}'.")

    df = pd.read_csv(path)
    logger.info("Dataset carregado de %s: %d linhas x %d colunas", path, *df.shape)

    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {sorted(missing)}")

    return clean(df)