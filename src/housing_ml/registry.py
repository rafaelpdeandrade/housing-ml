import json
from housing_ml.logging_config import setup_logging
from dataclasses import dataclass
from pathlib import Path
from housing_ml.config import Settings

import joblib

logger = setup_logging(level="INFO", fmt="text", log_file=Path(Settings.ARTIFACTS_DIR / "Logs" / "registry.log"))

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


@dataclass(frozen=True)
class ModelArtifacts:
    model: object
    metadata: dict
   

def save_artifact(model, metadata: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    # Modelo
    joblib.dump(value=model, filename=directory / MODEL_FILE)
    # Metadata
    (directory / METADATA_FILE).write_text(json.dumps(metadata, indent=2))

    logger.info("Artefato salvo em %s (versão %s)", directory, metadata["version"])

def load_artifact(directory: Path) -> ModelArtifacts:
    model_path = directory / MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em '{model_path}'. Rode o treino primeiro")

    metadata = json.loads((directory / METADATA_FILE).read_text())

    logger.info("Artefato carregado de %s (versão %s)", directory, metadata["version"])

    return ModelArtifacts(
        model = joblib.load(model_path),
        metadata = metadata
    )