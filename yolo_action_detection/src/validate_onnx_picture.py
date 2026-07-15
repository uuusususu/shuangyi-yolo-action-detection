"""Validate a pure ONNX OBB model on a static picture.

This entrypoint intentionally avoids camera/UI dependencies so ONNX geometry can
be verified before live realtime testing.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Type

import cv2
import numpy as np

from config import ConfigManager
from pcb_inspection.engine import MultiPcbInspectionEngine
from pcb_inspection.models import PcbInspectionConfig, PcbResult
from yolo_runtime.onnx_obb_processor import OnnxObbProcessor
from yolo_runtime.yolo_result_models import DetectionOverlayState, ObbDetection


def evaluate_first_category_region(
    detections: list[ObbDetection],
    image_size: tuple[int, int],
    config,
    frame_repetitions: Optional[int] = None,
) -> Dict[str, object]:
    """用生产区域引擎验证父类优先、区域内子类别数量和最终结果。"""
    if not getattr(config, "first_category_region_check_enabled", False):
        return {
            "ok": False,
            "status": "disabled",
            "error": "首类别区域检查未启用",
            "parents": [],
        }

    region_config = PcbInspectionConfig.from_first_category_config(config)
    repetitions = frame_repetitions
    if repetitions is None:
        repetitions = max(region_config.pass_stable_frames, region_config.fail_stable_frames)
    repetitions = max(1, int(repetitions))

    engine = MultiPcbInspectionEngine(region_config)
    events = []
    for _ in range(repetitions):
        events.extend(engine.update(detections, image_size=image_size))

    parents = []
    for track_id in sorted(engine.current_parent_ids):
        state = engine.pcb_states[track_id]
        parents.append({
            "track_id": track_id,
            "status": state.status.value,
            "result": state.result.value,
            "completed_classes": sorted(state.completed_classes),
            "observed_counts": {
                name: slot.observed_count
                for name, slot in state.last_slot_states.items()
            },
            "required_counts": {
                name: slot.required_count
                for name, slot in state.last_slot_states.items()
            },
            "slot_statuses": {
                name: slot.status.value
                for name, slot in state.last_slot_states.items()
            },
        })

    if not parents:
        status = "no_parent"
    elif any(parent["result"] == PcbResult.FAIL.value for parent in parents):
        status = "ng"
    elif all(parent["result"] == PcbResult.PASS.value for parent in parents):
        status = "pass"
    else:
        status = "waiting"

    errors = {
        "pass": "",
        "no_parent": f"未识别到父类: {region_config.pcb_class_name}",
        "ng": "父类区域内子类别数量异常",
        "waiting": "父类已识别，但子步骤尚未全部完成",
    }

    return {
        "ok": status == "pass",
        "status": status,
        "error": errors[status],
        "parent_class": region_config.pcb_class_name,
        "child_required_counts": dict(region_config.component_required_counts),
        "parent_detection_count": len(parents),
        "frame_repetitions": repetitions,
        "parents": parents,
        "events": [
            {
                "track_id": result.track_id,
                "result": result.result.value,
                "observed_counts": dict(result.observed_counts),
                "required_counts": dict(result.required_counts),
            }
            for result in events
        ],
    }


def run_validation(
    image_path: Path | str,
    model_path: Path | str,
    output_dir: Path | str,
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    processor_factory: Optional[Type[OnnxObbProcessor] | Callable[..., object]] = None,
    region_config_path: Optional[Path | str] = None,
    region_frame_repetitions: Optional[int] = None,
) -> Dict[str, object]:
    """Run ONNX OBB inference on one image and write summary + annotated image."""
    image_path = Path(image_path)
    model_path = Path(model_path)
    output_dir = Path(output_dir)

    if not model_path.exists():
        return {
            "ok": False,
            "error": f"模型文件不存在: {model_path}",
            "model_path": str(model_path),
            "image_path": str(image_path),
        }
    if not image_path.exists():
        return {
            "ok": False,
            "error": f"图片文件不存在: {image_path}",
            "model_path": str(model_path),
            "image_path": str(image_path),
        }

    frame = read_image(image_path)
    if frame is None:
        return {
            "ok": False,
            "error": f"图片读取失败: {image_path}",
            "model_path": str(model_path),
            "image_path": str(image_path),
        }

    factory = processor_factory or OnnxObbProcessor
    processor = factory(
        model_path=str(model_path),
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
    t0 = time.perf_counter()
    processor.load()
    overlay = processor.process_frame(frame, source_frame_id=1, round_id=0)
    total_latency_ms = (time.perf_counter() - t0) * 1000

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    annotated_path = output_dir / f"{stem}.onnx_obb_annotated.jpg"
    summary_path = output_dir / f"{stem}.onnx_obb_summary.json"

    annotated = draw_overlay(frame.copy(), overlay)
    write_image(annotated_path, annotated)

    summary: Dict[str, object] = {
        "ok": overlay.error == "",
        "error": overlay.error,
        "model_path": str(model_path),
        "image_path": str(image_path),
        "annotated_path": str(annotated_path),
        "summary_path": str(summary_path),
        "task_type": overlay.task_type,
        "class_names": processor.get_class_names(),
        "image_shape": [int(frame.shape[0]), int(frame.shape[1]), int(frame.shape[2])],
        "load_plus_infer_latency_ms": total_latency_ms,
        "infer_latency_ms": overlay.latency_ms,
        "detection_count": len(overlay.detections),
        "detections": [det.to_dict() for det in overlay.detections],
    }
    if region_config_path is not None:
        region_config = ConfigManager()
        region_config.load(region_config_path)
        region_summary = evaluate_first_category_region(
            overlay.detections,
            image_size=(int(frame.shape[1]), int(frame.shape[0])),
            config=region_config,
            frame_repetitions=region_frame_repetitions,
        )
        summary["first_category_region"] = region_summary
        if not region_summary["ok"]:
            summary["ok"] = False
            if not summary["error"]:
                summary["error"] = region_summary["error"]
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def draw_overlay(frame: np.ndarray, overlay: DetectionOverlayState) -> np.ndarray:
    """Draw OBB polygons on an image for manual review."""
    for det in overlay.detections:
        if len(det.polygon) < 4:
            continue
        pts = np.array(det.polygon, dtype=np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 128), 2)
        x, y = pts[0]
        label = f"{det.label} {det.conf:.2f}"
        cv2.putText(
            frame,
            label,
            (int(x), max(15, int(y) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 128),
            1,
            cv2.LINE_AA,
        )
    if overlay.error:
        cv2.putText(
            frame,
            overlay.error[:80],
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


def read_image(path: Path | str) -> Optional[np.ndarray]:
    """Read image from paths that may contain non-ASCII characters on Windows."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path | str, image: np.ndarray) -> bool:
    """Write image to paths that may contain non-ASCII characters on Windows."""
    path = Path(path)
    ext = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ONNX OBB model on one picture.")
    parser.add_argument("--image", required=True, help="Path to the test picture.")
    parser.add_argument(
        "--model",
        default="config/best.onnx",
        help="Path to ONNX model. Defaults to config/best.onnx.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/onnx_picture_validation",
        help="Directory for summary JSON and annotated image.",
    )
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="Rotated NMS IoU threshold.")
    parser.add_argument(
        "--region-config",
        help="启用首类别区域业务验证时使用的配置 JSON。",
    )
    parser.add_argument(
        "--region-frames",
        type=int,
        help="静态检测结果重复送入区域引擎的帧数；默认使用 PASS/FAIL 稳定帧最大值。",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = run_validation(
        image_path=args.image,
        model_path=args.model,
        output_dir=args.output_dir,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        region_config_path=args.region_config,
        region_frame_repetitions=args.region_frames,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
