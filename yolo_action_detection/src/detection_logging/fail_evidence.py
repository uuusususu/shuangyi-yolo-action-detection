"""FAIL 证据保存。"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2

logger = logging.getLogger("yolo_action_detection.runtime")


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
    source_roi: dict[str, int] | None = None
    actual_count: int | None = None
    required_count: int | None = None
    fail_retry_count: int | None = None
    ng_reason: str = ""
    boundary_frame: int = 0
    decision_frame: int = 0
    seen_parts: list[str] = field(default_factory=list)
    missing_parts: list[str] = field(default_factory=list)
    model_sha256: str = ""
    # 连续动作使用采集单调时钟；timestamp 始终为存档日期所用的 Unix 墙钟。
    action_start_time: float | None = None
    action_end_time: float | None = None
    action_decision_time: float | None = None


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
                logger.error("fail_evidence=encode_failed folder=%s", folder)
                return None
            image_path.write_bytes(encoded.tobytes())

            metadata = self._metadata(context, timestamp, image_path.name)
            (folder / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return folder
        except Exception:
            # 调用方只拿到 None，具体原因记在这里。
            logger.exception("fail_evidence=save_failed base_dir=%s", self.base_dir)
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
        metadata = {
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
        if context.source_roi is not None:
            metadata["source_roi"] = _json_safe_dict(context.source_roi)
        if context.actual_count is not None:
            metadata["actual_count"] = int(context.actual_count)
        if context.required_count is not None:
            metadata["required_count"] = int(context.required_count)
        if context.fail_retry_count is not None:
            metadata["fail_retry_count"] = int(context.fail_retry_count)
        if context.ng_reason:
            metadata["ng_reason"] = str(context.ng_reason)
        if context.boundary_frame:
            metadata["boundary_frame"] = int(context.boundary_frame)
        if context.decision_frame:
            metadata["decision_frame"] = int(context.decision_frame)
        if context.seen_parts:
            metadata["seen_parts"] = list(context.seen_parts)
        if context.missing_parts:
            metadata["missing_parts"] = list(context.missing_parts)
        if context.model_sha256:
            metadata["model_sha256"] = str(context.model_sha256)
        action_times = {
            "action_start_time": context.action_start_time,
            "action_end_time": context.action_end_time,
            "action_decision_time": context.action_decision_time,
        }
        if any(value is not None for value in action_times.values()):
            metadata["action_time_clock"] = "monotonic"
            for key, value in action_times.items():
                if value is not None:
                    metadata[key] = float(value)
        return metadata


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
