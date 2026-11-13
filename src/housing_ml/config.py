from functools import lru_cache
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HOUSING_", env_file=".env", extra="ignore")

    BASE_PATH: Path = Path(__file__).resolve().parent.parent.parent        # Raiz do projeto 'housing-ml'
    DATA_PATH: Path = BASE_PATH / "data" / "raw" / "housing.csv"
    ARTIFACTS_DIR: Path = BASE_PATH / "artifacts"

    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5
    RANDOM_STATE: int = 42

    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    LOG_FORMAT: Literal["text", "json"] = "text"
    LOG_FILE: Path | None = None        # arquivo opcional para a API; o treino usa artifacts/runs/<versao>/


@lru_cache
def get_settings() -> Settings:
    return Settings()
