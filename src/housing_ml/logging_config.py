import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from logging.config import dictConfig
from pathlib import Path

# Identificador da requisição atual; "-" quando não há requisição (ex: treino)
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

TEXT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s | [%(request_id)s] %(message)s"
_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "request_id", "color_message"}

class RequestIdFilter(logging.Filter):
    """ Injeta o 'request_id' em todo log, sem que o código de negócio precise passá-lo. """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True

class JsonFormatter(logging.Formatter):
    """ Uma linha JSON por log; campos passados em `extra=` viram chaves do JSON. """

    converter = time.gmtime     # timestamps em UTC

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": record.request_id,
            "message": record.getMessage()
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS})
        # Se houver erro de exeção, captura o rastreamento completo 
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dump(payload, ensure_ascii=False, default=str)

def setup_logging(level: str = "INFO", fmt: str = "text", log_file: Path | None = None) -> None:
    """ Configura o 'logging' da aplicação. Deve ser chamada uma vez por ponto de entrada. """

    formatter = "json" if fmt == "json" else "text"
    handlers: dict[str, dict] = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": formatter,
            "filters": ["request_id"]
        }
    }

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(log_file),
            "maxBytes": 5_000_000,
            "backupCount": 3,
            "encoding": "utf-8",
            "formatter": formatter,
            "filters": ["request_id"]
        }

    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"request_id": {"()": RequestIdFilter}},
        "formatters": {"text": {"format": TEXT_FORMAT}, "json": {"()": JsonFormatter}},
        "handlers": handlers,
        "root": {"level": level, "handlers": list(handlers)},
        "loggers": {
            # logs do uvicorn passam a usar nossos handlers (via root)
            "uvicorn": {"handlers": [], "level": level, "propagate": True},
            # o access log do uvicorn é substituído pelo nosso middleware
            "uvicorn.access": {"hanlders": [], "propagate": False}
        },
    })      # fmt: sskip

@contextmanager
def log_step(logger: logging.Logger, step: str) -> Iterator[None]:
    """ Loga início, fim e duração de uma etapa; em caso de erro, loga a stack trace. """
    logger.info("%s: iniciando", step)

    start = time.perf_counter()
    try:
        yield
    except Exception:
        logger.exception("%s: falhou após %.2fs", step, time.perf_counter() - start)
        raise

    duration = time.perf_counter() - start
    logger.info("%s: concluído em %.2fs", step, duration, extra={"step": step, "duration_s": round(duration, 3)},
                )

