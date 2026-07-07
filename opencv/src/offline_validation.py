"""离线图片验证入口。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from config import ConfigManager
from mediapipe_frame_processor import MediaPipeGestureProcessor as FrameProcessor


DEFAULT_SNAPSHOT_NAME = "cam-Snapshot-20260408-143156-780-6003005650747.jpg"


def _is_validation_passed(task_type: str, detections: list[dict], classification: dict, matched: list[str]) -> bool:
    task = str(task_type or "").strip().lower()
    has_spatial = any(
        (det.get("polygon") is not None)
        or (len(det.get("box", [])) >= 4)
        for det in detections
    )
    has_keypoints = any(
        isinstance(det.get("keypoints"), dict) and bool(det["keypoints"].get("points"))
        for det in detections
    )
    has_classification = bool(str(classification.get("label", "")).strip()) and float(
        classification.get("conf", 0.0)
    ) > 0.0

    if task == "mediapipe_gesture":
        return bool(matched)
    if task == "classify":
        return has_classification
    if task == "pose":
        return has_keypoints or bool(detections)
    if task in {"segment", "detect", "obb"}:
        return bool(detections) and has_spatial and bool(matched)
    return bool(detections) or has_classification


def _read_image(image_path: Path):
    """兼容 Windows 中文路径读取图片。"""
    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def validate_image(
    config: ConfigManager,
    image_path: str | Path,
    *,
    output_dir: str | Path,
) -> Dict[str, object]:
    """对单张图片执行与实时链路一致的离线验证。"""
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"测试图片不存在: {image_path}")

    frame = _read_image(image_path)
    if frame is None or frame.size == 0:
        raise ValueError(f"无法读取测试图片: {image_path}")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = FrameProcessor(config)
    overlay_state = processor.process_overlay(frame.copy(), source_frame_id=1, round_id=1)
    rendered = processor.render_overlay(frame.copy(), overlay_state)
    snapshot = processor.get_last_inference_snapshot()

    annotated_path = output_dir / f"{image_path.stem}.annotated.png"
    summary_path = output_dir / f"{image_path.stem}.summary.json"
    cv2.imwrite(str(annotated_path), rendered)

    detections = list(snapshot.get("detections", []) or [])
    matched = list(snapshot.get("matched_category_names", []) or [])
    classification = dict(snapshot.get("classification", {}) or {})
    task_type = str(snapshot.get("task_type", getattr(config, "model_task", "obb")) or "obb")
    passed = _is_validation_passed(task_type, detections, classification, matched)

    summary: Dict[str, object] = {
        "image_path": str(image_path),
        "annotated_path": str(annotated_path),
        "model_path": str(snapshot.get("model_path", config.get_model_path())),
        "task_type": task_type,
        "matched_category_names": matched,
        "detections": detections,
        "classification": classification,
        "status": str(snapshot.get("status", "unknown")),
        "passed": bool(passed),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def validate_default_snapshot(
    config: Optional[ConfigManager] = None,
    *,
    project_root: str | Path,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, object]:
    """验证项目根目录下的固定快照样本。"""
    root = Path(project_root).resolve()
    cfg = config or ConfigManager()
    image_path = root / DEFAULT_SNAPSHOT_NAME
    if output_dir is None:
        output_dir = root / "opencv" / "offline_validation"
    return validate_image(cfg, image_path, output_dir=output_dir)
