"""Persistent diagnostics for portable runtime failures."""
from __future__ import annotations

import hashlib
import logging
import os
import platform
import sys
import tempfile
from pathlib import Path


LOGGER_NAME = "yolo_action_detection.runtime"
LOG_FILENAME = "startup_diagnostics.log"


def get_runtime_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def get_runtime_log_path() -> Path | None:
    for handler in get_runtime_logger().handlers:
        filename = getattr(handler, "baseFilename", None)
        if filename:
            return Path(filename)
    return None


def close_runtime_logger() -> None:
    logger = get_runtime_logger()
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _candidate_log_paths(base_dir: str | Path) -> list[Path]:
    candidates = [Path(base_dir) / "logs" / LOG_FILENAME]
    fallback_root = Path(
        os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    ) / "YOLOActionDetection"
    fallback = fallback_root / "logs" / LOG_FILENAME
    if fallback != candidates[0]:
        candidates.append(fallback)
    return candidates


def configure_runtime_logger(base_dir: str | Path) -> Path | None:
    """Create a UTF-8 log containing only the latest application run."""
    close_runtime_logger()
    logger = get_runtime_logger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for log_path in _candidate_log_paths(base_dir):
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(
                log_path,
                mode="w",
                encoding="utf-8",
            )
        except OSError:
            continue
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s.%(msecs)03d %(levelname)s %(threadName)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.info("diagnostics_log=%s", log_path)
        return log_path

    logger.addHandler(logging.NullHandler())
    return None


def install_unhandled_exception_hook() -> None:
    previous_hook = sys.excepthook

    def log_exception(exc_type, exc_value, exc_traceback) -> None:
        get_runtime_logger().critical(
            "unhandled_exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        previous_hook(exc_type, exc_value, exc_traceback)

    sys.excepthook = log_exception


def log_startup_context(
    *,
    base_dir: str | Path,
    resource_dir: str | Path,
    config_path: str | Path,
    frozen: bool,
) -> None:
    logger = get_runtime_logger()
    config = Path(config_path)
    logger.info("session_start")
    logger.info(
        "runtime frozen=%s executable=%s cwd=%s",
        frozen,
        sys.executable,
        Path.cwd(),
    )
    logger.info(
        "platform system=%s release=%s machine=%s processor=%s",
        platform.system(),
        platform.release(),
        platform.machine(),
        platform.processor() or "unknown",
    )
    logger.info(
        "python version=%s architecture=%s",
        platform.python_version(),
        platform.architecture()[0],
    )
    logger.info("base_dir=%s", Path(base_dir))
    logger.info("resource_dir=%s", Path(resource_dir))
    logger.info("config_path=%s config_exists=%s", config, config.is_file())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log_model_context(model_path: str | Path, *, include_hash: bool = False) -> None:
    logger = get_runtime_logger()
    path = Path(model_path)
    exists = path.is_file()
    logger.info("model_path=%s model_exists=%s", path, exists)
    if exists:
        try:
            logger.info("model_size_bytes=%s", path.stat().st_size)
            if include_hash:
                logger.info("model_sha256=%s", _sha256(path))
        except OSError:
            logger.exception("model_metadata=failed")

    if path.suffix.lower() != ".onnx":
        return
    try:
        import onnxruntime as ort

        logger.info(
            "onnxruntime_import=ok version=%s module=%s providers=%s",
            getattr(ort, "__version__", "unknown"),
            getattr(ort, "__file__", "unknown"),
            ort.get_available_providers(),
        )
    except Exception:
        logger.exception("onnxruntime_import=failed")
