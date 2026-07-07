"""JSONL 检测日志模块。"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional


class DetectionSessionLogger:
    """每帧检测日志记录器。"""

    def __init__(self, log_dir: str = "logs/detections") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._session_id = ""

    def start_session(self, session_id: Optional[str] = None) -> str:
        self._session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        log_path = self._log_dir / f"{self._session_id}.jsonl"
        self._file = open(log_path, "w", encoding="utf-8")
        return self._session_id

    def log_frame(self, record: Dict) -> None:
        if self._file:
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
