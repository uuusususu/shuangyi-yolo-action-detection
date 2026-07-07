"""Detection session logger for structured JSONL output."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class DetectionSessionLogger:
    """Append-only JSONL logger with explicit session lifecycle."""

    def __init__(
        self,
        *,
        base_dir: Path,
        log_dir: str = "logs/detections",
        flush_interval: int = 30,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._log_dir = Path(log_dir)
        self._flush_interval = max(1, int(flush_interval))

        self._lock = threading.Lock()
        self._file = None
        self._line_count = 0
        self._session_id = ""
        self._session_path: Optional[Path] = None

    @property
    def session_path(self) -> Optional[Path]:
        return self._session_path

    def _resolved_log_dir(self) -> Path:
        if self._log_dir.is_absolute():
            return self._log_dir
        return (self._base_dir / self._log_dir).resolve()

    def start_session(self, metadata: Optional[Dict[str, Any]] = None) -> Path:
        with self._lock:
            if self._file is not None:
                self.stop_session({"reason": "restart_session"})

            resolved_dir = self._resolved_log_dir()
            resolved_dir.mkdir(parents=True, exist_ok=True)

            self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self._session_path = resolved_dir / f"detection_session_{self._session_id}.jsonl"
            self._file = self._session_path.open("a", encoding="utf-8")
            self._line_count = 0

            payload = {
                "type": "session_start",
                "timestamp": time.time(),
                "session_id": self._session_id,
                "metadata": metadata or {},
            }
            self._write_payload(payload, force_flush=True)
            return self._session_path

    def log_frame(self, record: Dict[str, Any]) -> bool:
        with self._lock:
            if self._file is None:
                return False

            payload = {
                "type": "frame",
                "timestamp": time.time(),
                **record,
            }
            self._write_payload(payload)
            return True

    def stop_session(self, summary: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            if self._file is None:
                return

            payload = {
                "type": "session_end",
                "timestamp": time.time(),
                "session_id": self._session_id,
                "summary": summary or {},
            }
            self._write_payload(payload, force_flush=True)

            self._file.close()
            self._file = None
            self._line_count = 0
            self._session_id = ""

    def _write_payload(self, payload: Dict[str, Any], *, force_flush: bool = False) -> None:
        if self._file is None:
            return

        self._file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._line_count += 1

        if force_flush or (self._line_count % self._flush_interval == 0):
            self._file.flush()

