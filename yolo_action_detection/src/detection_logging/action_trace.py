"""产线动作追溯。

启动日志只保留这一次运行的环境。这份文件按行留下每次采样、切换确认和
事件被接收或拒收的原因，方便事后解释直通率。
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path


LOGGER_NAME = "yolo_action_detection.action_trace"
LOG_FILENAME = "action_trace.log"
_MAX_BYTES = 20 * 1024 * 1024


def configure_action_trace(base_dir: str | Path) -> Path | None:
    """在程序目录 logs 下追加追溯；写不了时改到本机用户目录。"""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for log_path in _candidate_log_paths(base_dir):
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            _rotate_if_large(log_path)
            handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        except OSError:
            continue
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s.%(msecs)03d %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.info("session_start path=%s", log_path)
        handler.flush()
        return log_path

    logger.addHandler(logging.NullHandler())
    return None


def trace(kind: str, **fields) -> None:
    """写一行追溯。未配置或写失败时不影响判定。"""
    logger = logging.getLogger(LOGGER_NAME)
    if not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
        return
    parts = [str(kind)]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={str(value).replace(chr(10), ' ')}")
    try:
        logger.info(" ".join(parts))
        for handler in logger.handlers:
            handler.flush()
    except Exception:
        return


def _candidate_log_paths(base_dir: str | Path) -> list[Path]:
    candidates = [Path(base_dir) / "logs" / LOG_FILENAME]
    fallback_root = Path(
        os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    ) / "YOLOActionDetection"
    fallback = fallback_root / "logs" / LOG_FILENAME
    if fallback != candidates[0]:
        candidates.append(fallback)
    return candidates


def _rotate_if_large(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < _MAX_BYTES:
        return
    backup = path.with_name("action_trace.1.log")
    if backup.exists():
        backup.unlink()
    path.replace(backup)
