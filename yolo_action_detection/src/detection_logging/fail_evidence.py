"""FAIL 证据保存。"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2


@dataclass(frozen=True)
class FailEvidenceContext:
    """一次 ACTION_NG 事件的证据上下文。"""

    round_id: int
    action_ng_step: int
    step_name: str
    source_frame_id: int
    detections: list[Any] = field(default_factory=list)
    model_path: str = ""
    timestamp: float = 0.0


class FailEvidenceSaver:
    """按 FAIL 事件保存图片和元数据。"""

    def __init__(self, *, enabled: bool = True, base_dir: str | Path = "outputs/evidence") -> None:
        self.enabled = bool(enabled)
        self.base_dir = Path(base_dir)

    def save_fail_event(self, frame, context: FailEvidenceContext) -> Path | None:
        if not self.enabled or frame is None:
            return None

        try:
            timestamp = float(context.timestamp or time.time())
            event_time = datetime.fromtimestamp(timestamp)
            folder = self._build_event_dir(event_time, context)
            folder.mkdir(parents=True, exist_ok=False)

            image_path = folder / "frame.jpg"
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                return None
            image_path.write_bytes(encoded.tobytes())

            metadata = self._metadata(context, timestamp, image_path.name)
            (folder / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return folder
        except Exception:
            return None

    def _build_event_dir(self, event_time: datetime, context: FailEvidenceContext) -> Path:
        day = event_time.strftime("%Y%m%d")
        clock = event_time.strftime("%H%M%S")
        step_no = int(context.action_ng_step) + 1
        name = f"NG_{clock}_round{int(context.round_id)}_step{step_no}"
        if context.source_frame_id:
            name = f"{name}_frame{int(context.source_frame_id)}"
        if context.step_name:
            name = f"{name}_{_safe_name(context.step_name)}"
        return self.base_dir / day / name

    def _metadata(
        self,
        context: FailEvidenceContext,
        timestamp: float,
        image_name: str,
    ) -> dict[str, Any]:
        return {
            "event": "ACTION_NG",
            "round_id": int(context.round_id),
            "action_ng_step": int(context.action_ng_step),
            "step_no": int(context.action_ng_step) + 1,
            "step_name": str(context.step_name),
            "source_frame_id": int(context.source_frame_id),
            "model_path": str(context.model_path),
            "timestamp": timestamp,
            "created_at": datetime.fromtimestamp(timestamp).isoformat(timespec="seconds"),
            "image": image_name,
            "detections": _serialize_detections(context.detections),
        }


def _serialize_detections(detections: Iterable[Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for det in detections or []:
        if hasattr(det, "to_dict"):
            try:
                items.append(det.to_dict())
                continue
            except Exception:
                pass
        if isinstance(det, dict):
            items.append(_json_safe_dict(det))
            continue
        items.append({"value": str(det)})
    return items


def _json_safe_dict(data: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in data.items():
        safe[str(key)] = _json_safe_value(value)
    return safe


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return _json_safe_dict(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    return str(value)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", value.strip())
    return cleaned[:32] or "step"
