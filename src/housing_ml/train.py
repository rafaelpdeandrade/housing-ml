import logging
from pathlib import Path
from datetime import UTC, datetime

import pandas as pd
import sklearn
from sklearn.model_selection import KFold, cross_validate, train_test_split

from housing_ml import __version__
from housing_ml.config import Settings, get_settings
from housing_ml.data import load_dataset
from housing_ml.evaluate import evaluate
from housing_ml.features import FEATURES, TARGET
from housing_ml.logging_config import setup_logging, log_step
from housing_ml.model import build_model
from housing_ml.registry import save_artifact

logger = setup_logging(level="INFO", fmt="text", log_file=Path(Settings.ARTIFACTS_DIR / "Logs" / "train.log")


def new_version() -> str:
    return datetime.now(UTC).strftime("%Y%m%d%H%M%S")

def train_model(df: pd.DataFrame, settings: Settings, version: str | None = None):
    version = version or new_version()
    logger.info("Parâmetros: test_size=%s, cv_folds=%s, random_state=%s", settings.TEST_SIZE, settings.CV_FOLDS, settings.RANDOM_STATE)

    # Divisão dos dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X=df[FEATURES], y=df[TARGET], test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE)
    logger.info("Split: treino=%d, teste=%d", len(X_train), len(X_test))

    # Cross validation
    #monitorar o tempo de execução com: log_step
    with log_step(logger, "Validação cruzada"):
        cv = cross_validate(
            estimator=build_model(),
            X = X_train,
            y = y_train,
            cv = KFold(settings.CV_FOLDS, shuffle=True, random_state = settings.RANDOM_STATE),
            scoring = {"rmse": "neg_root_mean_squared_error",
                       "r2": "r2"},
            n_jobs = -1     # usar o máximo de processos paralelos
        )

    # Métricas
    cv_metrics = {
        "rmse_mean": float(-cv["test_rmse"].mean()),    # -: inverte o sinal para avaliação humana de forma correta
        "rmse_std": float(cv["test_rmse"].std()),
        "r2_mean": float(cv["test_r2"].mean()),
        "r2_std": float(cv["test_r2"].std())
    }
    logger.info("Métricas CV: %s", cv_metrics)

    # Treino
    #monitorar o tempo de execução com: log_step
    with log_step(logger, "Treino final"):
        model = build_model().fit(X_train, y_train)

    # Teste / Predict
    with log_step(logger, "Avaliação do teste"):
        test_metrics = evaluate(model=model, X=X_test, y=y_test)
    logger.info("Métricas teste (predict): %s", test_metrics)

    # Metadata
    metadata = {
        "version": version,
        "package_version": __version__,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(UTC).isoformat(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": FEATURES,
        "target": TARGET,
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics
    }

    return model, metadata

def main() -> None:
    settings = get_settings()
    version = new_version()
    run_dir = settings.ARTIFACTS_DIR / "runs" / version                             # criando uma sub-pasta 'runs' e version (já em string)
    
    logger = setup_logging(settings.LOG_LEVEL, settings.LOG_FORMAT, run_dir / "train.log")   # criando um logger
    logger.info("Treino iniciado (versão %s)", version)

    # Carregar dataset
    #monitorar o tempo de execução com: log_step
    with log_step(logger, "Carga do dataset"):
        df = load_dataset(settings.DATA_PATH)

    model, metadata = train_model(df=df, settings=settings, version=version)

    # Salvar artefatos
    #monitorar o tempo de execução com: log_step
    with log_step(logger, "Salvamento do artefato"):
        save_artifact(model=model, metadata=metadata, directory=settings.ARTIFACTS_DIR)

    logger.info("Treino finalizado (versão %s)", version)

    
if __name__ == "__main__":
    main()