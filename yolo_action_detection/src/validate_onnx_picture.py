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
from roi_inspection.engine import RoiInspectionEngine
from roi_inspection.geometry import NormalizedRect
from roi_inspection.models import InspectionOutcome, RoiInspectionConfig, RoundPhase
from yolo_runtime.onnx_obb_processor import OnnxObbProcessor
from yolo_runtime.yolo_result_models import DetectionOverlayState, ObbDetection


def evaluate_roi_inspection(
    detections: list[ObbDetection],
    image_size: tuple[int, int],
    config,
    model_class_names=None,
    frame_repetitions: Optional[int] = None,
) -> Dict[str, object]:
    """使用与实时检测相同的配置校验、ROI 几何和轮次引擎。"""
    if not getattr(config, "roi_inspection_enabled", False):
        return {
            "ok": False,
            "status": "disabled",
            "error": "ROI 定位检测未启用",
        }
    errors = config.inspection_validation_errors(model_class_names)
    if errors:
        return {
            "ok": False,
            "status": "invalid_config",
            "error": "；".join(errors),
            "config_errors": errors,
        }

    roi_config = RoiInspectionConfig.from_config(config)
    repetitions = frame_repetitions
    if repetitions is None:
        repetitions = max(roi_config.pass_stable_frames, roi_config.fail_stable_frames)
    repetitions = max(1, int(repetitions))

    engine = RoiInspectionEngine(roi_config)
    events = []
    for index in range(repetitions):
        events.extend(
            engine.update(
                detections,
                image_size=image_size,
                source_frame_id=index + 1,
                timestamp=float(index),
            )
        )
        if events:
            break

    snapshot = engine.snapshot
    if snapshot.result and snapshot.result.outcome == InspectionOutcome.PASS:
        status = "pass"
    elif snapshot.result and snapshot.result.outcome == InspectionOutcome.FAIL:
        status = "ng"
    elif snapshot.phase == RoundPhase.WAITING_FOR_LOCATOR:
        status = "waiting_locator"
    else:
        status = "inspecting"

    errors = {
        "pass": "",
        "waiting_locator": "等待模板匹配定位或识别 ROI 内出现目标",
        "ng": "识别 ROI 内待检类别数量异常",
        "inspecting": "已触发本轮，但稳定帧尚未达到 PASS/FAIL 阈值",
    }

    return {
        "ok": status == "pass",
        "status": status,
        "error": errors[status],
        "round_id": snapshot.round_id,
        "phase": snapshot.phase.value,
        "template_enabled": roi_config.template_enabled,
        "template_matched": snapshot.template_matched,
        "template_score": snapshot.template_score,
        "frame_repetitions": repetitions,
        "categories": {
            name: {
                "observed_count": state.observed_count,
                "required_count": state.required_count,
                "passed": state.passed,
                "match_streak": state.match_streak,
                "mismatch_streak": state.mismatch_streak,
            }
            for name, state in snapshot.category_states.items()
        },
        "events": [
            {
                "round_id": result.round_id,
                "result": result.outcome.value,
                "observed_counts": dict(result.actual_counts),
                "required_counts": dict(result.required_counts),
                "failed_categories": list(result.failed_categories),
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
    roi_config_path: Optional[Path | str] = None,
    roi_frame_repetitions: Optional[int] = None,
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
    if roi_config_path is not None:
        roi_config = ConfigManager()
        roi_config.load(roi_config_path)
        roi_summary = evaluate_roi_inspection(
            overlay.detections,
            image_size=(int(frame.shape[1]), int(frame.shape[0])),
            config=roi_config,
            model_class_names=processor.get_class_names().values(),
            frame_repetitions=roi_frame_repetitions,
        )
        summary["roi_inspection"] = roi_summary
        annotated = draw_roi_rects(annotated, roi_config)
        write_image(annotated_path, annotated)
        if not roi_summary["ok"]:
            summary["ok"] = False
            if not summary["error"]:
                summary["error"] = roi_summary["error"]
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


def draw_roi_rects(frame: np.ndarray, config) -> np.ndarray:
    """在静态验收图上绘制与实时界面一致的识别 ROI。"""
    if not getattr(config, "roi_inspection_enabled", False):
        return frame
    height, width = frame.shape[:2]
    for value, color, label in (
        (config.roi_inspection_rect, (87, 217, 126), "INSPECTION"),
    ):
        try:
            rect = NormalizedRect.from_mapping(value)
        except ValueError:
            continue
        x, y, rect_width, rect_height = rect.to_pixel_rect((width, height))
        p1 = (int(round(x)), int(round(y)))
        p2 = (int(round(x + rect_width)), int(round(y + rect_height)))
        cv2.rectangle(frame, p1, p2, color, 2)
        cv2.putText(frame, label, (p1[0] + 4, max(18, p1[1] + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
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
        "--roi-config",
        help="启用 ROI 定位业务验收时使用的配置 JSON。",
    )
    parser.add_argument(
        "--roi-frames",
        type=int,
        help="静态检测结果重复送入 ROI 引擎的帧数；默认使用 PASS/FAIL 稳定帧最大值。",
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
        roi_config_path=args.roi_config,
        roi_frame_repetitions=args.roi_frames,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
