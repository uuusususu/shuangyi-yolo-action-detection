"""YOLO 坐标诊断日志。"""
from __future__ import annotations

import json
import math
import queue
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from calibration.calibrator import (
    extract_tip_point,
    find_nearest_hole,
    judge_hole_zone,
)
from calibration.models import CalibrationTransform, HoleDefinition, TipPoint
from yolo_runtime.yolo_result_models import DetectionOverlayState, ObbDetection


SCHEMA_VERSION = "coordinate_log.v1"
OVERLAP_LIGHT_THRESHOLD = 0.05
OVERLAP_STRONG_THRESHOLD = 0.20


def resolve_coordinate_log_dir(log_dir: str | Path) -> Path:
    """把相对日志目录解析到 yolo_action_detection 下。"""
    path = Path(log_dir)
    if path.is_absolute():
        return path
    app_root = Path(__file__).resolve().parents[2]
    return app_root / path


def serialize_detection(
    detection: ObbDetection,
    calibrator: Optional[CalibrationTransform] = None,
) -> Dict[str, object]:
    """序列化单个检测结果，保留像素坐标和可选毫米坐标。"""
    center_mm = None
    if calibrator and calibrator.is_valid:
        try:
            center_mm = calibrator.pixel_to_mm(detection.center[0], detection.center[1])
        except Exception:
            center_mm = None
    return detection.to_dict(center_mm=center_mm)


def build_coordinate_frame_record(
    overlay: DetectionOverlayState,
    config,
    camera_params: Optional[Dict[str, object]] = None,
    pipeline_stats: Optional[Dict[str, object]] = None,
    image_size: Optional[Iterable[int]] = None,
    step_state=None,
) -> Dict[str, object]:
    """构造单帧坐标诊断记录。"""
    diagnostics: List[str] = []
    calibrator = _build_calibrator(config, diagnostics)
    holes = _build_holes(config, diagnostics)
    calibrated = bool(calibrator and calibrator.is_valid)
    tool_class = getattr(config, "tool_class_name", "扭力枪")
    frame_id = int(getattr(overlay, "source_frame_id", 0) or 0)

    detections = list(getattr(overlay, "detections", []) or [])
    serialized_detections = [serialize_detection(det, calibrator) for det in detections]
    tip_candidates = [
        serialize_detection(det, calibrator)
        for det in detections
        if det.label == tool_class
    ]
    overlap_analysis = build_overlap_analysis(detections, tool_class, _target_labels(config))

    tip, tip_diags = extract_tip_point(
        detections,
        tool_class,
        calibrator if calibrated else None,
        frame_id=frame_id,
        conf_threshold=0.0,
    )
    diagnostics.extend(tip_diags)
    if not calibrated:
        diagnostics.append("CalibrationMissing")
    if getattr(overlay, "error", ""):
        diagnostics.append(f"OverlayError({overlay.error})")

    current_step_index = _get_current_step_index(step_state)
    expected_hole = _find_hole(holes, current_step_index)
    nearest_hole = None
    nearest_distance = None
    distance_to_expected = None
    zone = None
    pass_candidate = False
    action_ng_candidate = False
    reason = ""

    if tip and calibrated:
        tip_mm = (tip.mm_x, tip.mm_y)
        nearest_hole, nearest_distance = find_nearest_hole(tip_mm, holes)
        if expected_hole:
            judgement = judge_hole_zone(tip_mm, expected_hole)
            distance_to_expected = float(judgement.distance_mm)
            zone = _enum_value(judgement.zone)
            pass_candidate = zone == "inside"
        if nearest_hole and expected_hole and nearest_hole.step_index != expected_hole.step_index:
            action_ng_candidate = nearest_distance is not None and nearest_distance <= nearest_hole.inner_radius_mm

    state_data = _serialize_step_state(step_state)
    round_result = state_data.get("round_result", "unknown")
    if round_result == "action_ng":
        reason = str(state_data.get("action_ng_reason", ""))
    elif pass_candidate:
        reason = "TipInsideExpectedHole"
    elif action_ng_candidate:
        reason = "TipInsideNonExpectedHole"

    record = {
        "schema_version": SCHEMA_VERSION,
        "frame_id": frame_id,
        "source_frame_id": frame_id,
        "timestamp": float(getattr(overlay, "timestamp", 0.0) or time.time()),
        "image_size": _normalize_image_size(image_size),
        "model_path": str(getattr(overlay, "model_path", "") or ""),
        "task_type": str(getattr(overlay, "task_type", "") or ""),
        "latency_ms": float(getattr(overlay, "latency_ms", 0.0) or 0.0),
        "status": str(getattr(overlay, "status", "") or ""),
        "error": str(getattr(overlay, "error", "") or ""),
        "camera_params": camera_params or {},
        "pipeline_stats": pipeline_stats or {},
        "detections": serialized_detections,
        "tip_candidates": tip_candidates,
        "overlap_analysis": overlap_analysis,
        "selected_tip": _tip_to_dict(tip, calibrated) if tip else None,
        "holes": [_hole_to_dict(h) for h in holes],
        "expected_hole": _hole_to_dict(expected_hole) if expected_hole else None,
        "nearest_hole": (
            _hole_to_dict(nearest_hole, nearest_distance) if nearest_hole else None
        ),
        "distance_to_expected_mm": distance_to_expected,
        "zone": zone,
        "pass_candidate": bool(pass_candidate),
        "action_ng_candidate": bool(action_ng_candidate),
        "reason": reason,
        "state": state_data,
        "diagnostics": _unique(diagnostics),
        "config_summary": _config_summary(config, calibrated, holes),
    }
    return record


def build_overlap_analysis(
    detections: List[ObbDetection],
    tool_class_name: str,
    target_labels: Optional[List[str]] = None,
) -> Dict[str, object]:
    """分析扭力枪 OBB 与步骤类别 OBB 的重合关系。"""
    target_label_set = {label for label in (target_labels or []) if label}
    tools = [det for det in detections if det.label == tool_class_name]
    if target_label_set:
        targets = [det for det in detections if det.label in target_label_set]
    else:
        targets = [det for det in detections if det.label != tool_class_name]

    pairs = []
    for tool in tools:
        for target in targets:
            polygon_iou = _polygon_iou(tool.polygon, target.polygon)
            bbox_iou = _bbox_iou(tool.box, target.box)
            center_distance = _center_distance_px(tool.center, target.center)
            pairs.append({
                "tool_label": tool.label,
                "tool_conf": float(tool.conf),
                "tool_track_id": tool.track_id,
                "tool_center_px": [float(tool.center[0]), float(tool.center[1])],
                "target_label": target.label,
                "target_class_id": int(target.class_id),
                "target_conf": float(target.conf),
                "target_track_id": target.track_id,
                "target_center_px": [float(target.center[0]), float(target.center[1])],
                "polygon_iou": float(polygon_iou),
                "bbox_iou": float(bbox_iou),
                "center_distance_px": float(center_distance),
                "overlap_level": _overlap_level(polygon_iou),
            })

    return {
        "tool_detected": bool(tools),
        "tool_count": len(tools),
        "target_count": len(targets),
        "pair_count": len(pairs),
        "overlap_pair_count": len([p for p in pairs if p["polygon_iou"] > 0]),
        "medium_or_strong_pair_count": len([
            p for p in pairs if p["polygon_iou"] >= OVERLAP_LIGHT_THRESHOLD
        ]),
        "strong_pair_count": len([
            p for p in pairs if p["polygon_iou"] >= OVERLAP_STRONG_THRESHOLD
        ]),
        "pairs": pairs,
    }


class CoordinateSessionLogger:
    """坐标日志会话写入器。"""

    def __init__(
        self,
        log_dir: str | Path = "logs/coordinate_sessions",
        max_queue: int = 1000,
        summary_enabled: bool = True,
        max_file_mb: int = 0,
        threaded: bool = True,
    ) -> None:
        self._log_dir = resolve_coordinate_log_dir(log_dir)
        self._max_queue = max(1, int(max_queue or 1))
        self._summary_enabled = bool(summary_enabled)
        self._max_file_bytes = max(0, int(max_file_mb or 0)) * 1024 * 1024
        self._threaded = threaded
        self._queue: Optional[queue.Queue] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._file = None
        self._session_id = ""
        self._log_path: Optional[Path] = None
        self._summary_path: Optional[Path] = None
        self._active = False
        self._last_error = ""
        self._submitted_count = 0
        self._written_count = 0
        self._dropped_count = 0
        self._summary = self._new_summary()
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self._active

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def log_path(self) -> str:
        return str(self._log_path or "")

    @property
    def summary_path(self) -> str:
        return str(self._summary_path or "")

    @property
    def last_error(self) -> str:
        return self._last_error

    def start_session(self, session_id: Optional[str] = None) -> str:
        """开始一个坐标日志会话。失败时返回空字符串并记录错误。"""
        if self._active:
            return self._session_id
        self._last_error = ""
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._session_id = self._unique_session_id(session_id)
            self._log_path = self._log_dir / f"{self._session_id}.ndjson"
            self._summary_path = self._log_dir / f"{self._session_id}.summary.json"
            self._file = open(self._log_path, "w", encoding="utf-8")
            self._queue = queue.Queue(maxsize=self._max_queue)
            self._stop_event.clear()
            self._submitted_count = 0
            self._written_count = 0
            self._dropped_count = 0
            self._summary = self._new_summary()
            self._summary.update({
                "session_id": self._session_id,
                "started_at": time.time(),
                "log_path": str(self._log_path),
                "summary_path": str(self._summary_path),
            })
            self._active = True
            if self._threaded:
                self._thread = threading.Thread(target=self._writer_loop, daemon=True)
                self._thread.start()
            return self._session_id
        except Exception as exc:
            self._last_error = f"坐标日志创建失败: {exc}"
            self._active = False
            self._close_file()
            return ""

    def log_frame(self, record: Dict[str, object]) -> bool:
        """提交单帧日志。队列满时丢弃并返回 False。"""
        if not self._active or self._queue is None:
            return False
        try:
            self._queue.put_nowait(dict(record))
            with self._lock:
                self._submitted_count += 1
            return True
        except queue.Full:
            self._increment_dropped("CoordinateLogQueueFull")
            return False

    def stop_session(self) -> Dict[str, object]:
        """停止会话并生成 summary。"""
        if not self._active:
            return self._final_summary()
        self._active = False
        self._stop_event.set()
        if self._threaded and self._thread:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                self._last_error = "坐标日志关闭超时，可能仍有少量记录未落盘"
        else:
            self._drain_pending()
        self._close_file()
        summary = self._final_summary()
        if self._summary_enabled and self._summary_path:
            try:
                self._summary_path.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                self._last_error = f"summary 写入失败: {exc}"
        return summary

    def close(self) -> Dict[str, object]:
        return self.stop_session()

    def get_status(self) -> Dict[str, object]:
        with self._lock:
            return {
                "active": self._active,
                "session_id": self._session_id,
                "log_path": str(self._log_path or ""),
                "summary_path": str(self._summary_path or ""),
                "submitted_count": self._submitted_count,
                "written_count": self._written_count,
                "dropped_count": self._dropped_count,
                "last_error": self._last_error,
            }

    def _writer_loop(self) -> None:
        while not self._stop_event.is_set() or (self._queue is not None and not self._queue.empty()):
            try:
                record = self._queue.get(timeout=0.1) if self._queue else None
            except queue.Empty:
                continue
            if record is None:
                continue
            try:
                self._write_record(record)
            finally:
                if self._queue:
                    self._queue.task_done()

    def _drain_pending(self) -> None:
        if not self._queue:
            return
        while True:
            try:
                record = self._queue.get_nowait()
            except queue.Empty:
                break
            self._write_record(record)
            self._queue.task_done()

    def _write_record(self, record: Dict[str, object]) -> None:
        if not self._file:
            self._increment_dropped("CoordinateLogFileClosed")
            return
        try:
            if self._max_file_bytes > 0 and self._file.tell() >= self._max_file_bytes:
                self._increment_dropped("CoordinateLogFileSizeLimit")
                return
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._file.flush()
            self._add_record_to_summary(record)
            with self._lock:
                self._written_count += 1
        except Exception as exc:
            self._increment_dropped(f"CoordinateLogWriteFailed({exc})")
            self._last_error = f"坐标日志写入失败: {exc}"

    def _increment_dropped(self, reason: str) -> None:
        with self._lock:
            self._dropped_count += 1
            self._summary["dropped_count"] = self._dropped_count
            self._last_error = reason

    def _unique_session_id(self, requested: Optional[str]) -> str:
        base = requested or f"coord_{time.strftime('%Y%m%d_%H%M%S')}"
        candidate = base
        index = 2
        while (self._log_dir / f"{candidate}.ndjson").exists():
            candidate = f"{base}_{index}"
            index += 1
        return candidate

    def _close_file(self) -> None:
        if self._file:
            try:
                self._file.close()
            except Exception as exc:
                self._last_error = f"坐标日志关闭失败: {exc}"
            self._file = None

    def _new_summary(self) -> Dict[str, object]:
        return {
            "schema_version": "coordinate_summary.v1",
            "session_id": "",
            "started_at": 0.0,
            "ended_at": 0.0,
            "log_path": "",
            "summary_path": "",
            "total_frames": 0,
            "valid_inference_frames": 0,
            "error_frames": 0,
            "dropped_count": 0,
            "labels": {},
            "no_tool_frames": 0,
            "duplicate_tool_frames": 0,
            "nearest_hole_distribution": {},
            "zone_distribution": {},
            "distance_stats_mm": {},
            "_distances": [],
            "_trigger_keys": set(),
            "triggers": [],
            "overlap_summary": {},
            "_overlap_streaks": {},
        }

    def _add_record_to_summary(self, record: Dict[str, object]) -> None:
        with self._lock:
            self._summary["total_frames"] += 1
            if record.get("error") or record.get("status") == "error":
                self._summary["error_frames"] += 1
            else:
                self._summary["valid_inference_frames"] += 1

            seen_labels = set()
            for det in record.get("detections", []) or []:
                label = str(det.get("label", ""))
                if not label:
                    continue
                seen_labels.add(label)
                label_stats = self._summary["labels"].setdefault(
                    label,
                    {"frame_count": 0, "total_count": 0, "_confs": []},
                )
                label_stats["total_count"] += 1
                label_stats["_confs"].append(float(det.get("conf", 0.0) or 0.0))
            for label in seen_labels:
                self._summary["labels"][label]["frame_count"] += 1

            if record.get("selected_tip") is None:
                self._summary["no_tool_frames"] += 1
            if len(record.get("tip_candidates", []) or []) > 1 or any(
                "DuplicateToolTipDetection" in str(d)
                for d in record.get("diagnostics", []) or []
            ):
                self._summary["duplicate_tool_frames"] += 1

            nearest = record.get("nearest_hole") or {}
            nearest_key = _hole_key(nearest)
            if nearest_key:
                _inc_dict(self._summary["nearest_hole_distribution"], nearest_key)

            zone = record.get("zone")
            if zone:
                _inc_dict(self._summary["zone_distribution"], str(zone))

            distance = record.get("distance_to_expected_mm")
            if isinstance(distance, (int, float)):
                self._summary["_distances"].append(float(distance))

            self._add_overlap_to_summary(record)

            state = record.get("state") or {}
            round_result = state.get("round_result")
            if round_result in ("pass", "action_ng"):
                key = (
                    state.get("round_id"),
                    round_result,
                    state.get("action_ng_step"),
                    state.get("action_ng_reason", ""),
                )
                if key not in self._summary["_trigger_keys"]:
                    self._summary["_trigger_keys"].add(key)
                    self._summary["triggers"].append({
                        "frame_id": record.get("frame_id"),
                        "round_id": state.get("round_id"),
                        "round_result": round_result,
                        "current_step_index": state.get("current_step_index"),
                        "action_ng_step": state.get("action_ng_step"),
                        "reason": state.get("action_ng_reason") or record.get("reason", ""),
                    })

    def _add_overlap_to_summary(self, record: Dict[str, object]) -> None:
        overlap = record.get("overlap_analysis") or {}
        pairs = overlap.get("pairs", []) or []
        best_by_label: Dict[str, Dict[str, object]] = {}
        for pair in pairs:
            label = str(pair.get("target_label", ""))
            if not label:
                continue
            current = best_by_label.get(label)
            if current is None or float(pair.get("polygon_iou", 0.0) or 0.0) > float(current.get("polygon_iou", 0.0) or 0.0):
                best_by_label[label] = pair

        known_labels = set(self._summary["overlap_summary"].keys()) | set(best_by_label.keys())
        stable_labels = set()
        for label, pair in best_by_label.items():
            polygon_iou = float(pair.get("polygon_iou", 0.0) or 0.0)
            bbox_iou = float(pair.get("bbox_iou", 0.0) or 0.0)
            center_distance = float(pair.get("center_distance_px", 0.0) or 0.0)
            stats = self._summary["overlap_summary"].setdefault(label, {
                "pair_frames": 0,
                "overlap_frames": 0,
                "medium_overlap_frames": 0,
                "strong_overlap_frames": 0,
                "max_polygon_iou": 0.0,
                "max_bbox_iou": 0.0,
                "min_center_distance_px": None,
                "_polygon_ious": [],
            })
            stats["pair_frames"] += 1
            stats["_polygon_ious"].append(polygon_iou)
            if polygon_iou > 0:
                stats["overlap_frames"] += 1
            if polygon_iou >= OVERLAP_LIGHT_THRESHOLD:
                stats["medium_overlap_frames"] += 1
                stable_labels.add(label)
            if polygon_iou >= OVERLAP_STRONG_THRESHOLD:
                stats["strong_overlap_frames"] += 1
            stats["max_polygon_iou"] = max(float(stats.get("max_polygon_iou", 0.0)), polygon_iou)
            stats["max_bbox_iou"] = max(float(stats.get("max_bbox_iou", 0.0)), bbox_iou)
            min_distance = stats.get("min_center_distance_px")
            if min_distance is None or center_distance < float(min_distance):
                stats["min_center_distance_px"] = center_distance

        for label in known_labels:
            stats = self._summary["overlap_summary"].setdefault(label, {
                "pair_frames": 0,
                "overlap_frames": 0,
                "medium_overlap_frames": 0,
                "strong_overlap_frames": 0,
                "max_polygon_iou": 0.0,
                "max_bbox_iou": 0.0,
                "min_center_distance_px": None,
                "_polygon_ious": [],
            })
            streak = self._summary["_overlap_streaks"].get(label, 0)
            if label in stable_labels:
                streak += 1
            else:
                streak = 0
            self._summary["_overlap_streaks"][label] = streak
            stats["longest_consecutive_overlap"] = max(
                int(stats.get("longest_consecutive_overlap", 0) or 0),
                streak,
            )

    def _final_summary(self) -> Dict[str, object]:
        with self._lock:
            summary = dict(self._summary)
            summary["ended_at"] = time.time()
            summary["dropped_count"] = self._dropped_count
            summary["labels"] = {
                label: _finalize_label_stats(stats)
                for label, stats in summary.get("labels", {}).items()
            }
            summary["distance_stats_mm"] = _distance_stats(summary.get("_distances", []))
            summary["overlap_summary"] = {
                label: _finalize_overlap_stats(stats)
                for label, stats in summary.get("overlap_summary", {}).items()
            }
            summary.pop("_distances", None)
            summary.pop("_trigger_keys", None)
            summary.pop("_overlap_streaks", None)
            return summary


def _build_calibrator(config, diagnostics: List[str]) -> Optional[CalibrationTransform]:
    cal_data = getattr(config, "calibration_points", []) or []
    if not cal_data:
        return None
    try:
        calibrator = CalibrationTransform.from_dict({"points": cal_data})
        if not calibrator.is_valid:
            diagnostics.append("CalibrationInvalid")
        return calibrator if calibrator.is_valid else None
    except Exception as exc:
        diagnostics.append(f"CalibrationInvalid({exc})")
        return None


def _target_labels(config) -> List[str]:
    labels = []
    for label in getattr(config, "category_names", []) or []:
        if label and str(label).strip():
            labels.append(str(label).strip())
    return labels


def _polygon_iou(poly_a, poly_b) -> float:
    if not poly_a or not poly_b or len(poly_a) < 3 or len(poly_b) < 3:
        return 0.0
    try:
        import cv2
        import numpy as np

        pa = np.array(poly_a, dtype=np.float32)
        pb = np.array(poly_b, dtype=np.float32)
        area_a = abs(float(cv2.contourArea(pa)))
        area_b = abs(float(cv2.contourArea(pb)))
        if area_a <= 0.0 or area_b <= 0.0:
            return 0.0
        inter_area, _ = cv2.intersectConvexConvex(pa, pb)
        union = area_a + area_b - float(inter_area)
        return float(inter_area) / union if union > 0.0 else 0.0
    except Exception:
        return 0.0


def _bbox_iou(box_a, box_b) -> float:
    if not box_a or not box_b or len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _center_distance_px(center_a, center_b) -> float:
    if not center_a or not center_b or len(center_a) < 2 or len(center_b) < 2:
        return 0.0
    return math.hypot(float(center_a[0]) - float(center_b[0]), float(center_a[1]) - float(center_b[1]))


def _overlap_level(polygon_iou: float) -> str:
    if polygon_iou <= 0.0:
        return "none"
    if polygon_iou < OVERLAP_LIGHT_THRESHOLD:
        return "light"
    if polygon_iou < OVERLAP_STRONG_THRESHOLD:
        return "medium"
    return "strong"


def _build_holes(config, diagnostics: List[str]) -> List[HoleDefinition]:
    holes = []
    for item in getattr(config, "holes", []) or []:
        try:
            if isinstance(item, HoleDefinition):
                holes.append(item)
            else:
                holes.append(HoleDefinition.from_dict(item))
        except Exception as exc:
            diagnostics.append(f"HoleConfigInvalid({exc})")
    return holes


def _find_hole(holes: List[HoleDefinition], step_index: int) -> Optional[HoleDefinition]:
    for hole in holes:
        if hole.enabled and hole.step_index == step_index:
            return hole
    return None


def _hole_to_dict(hole: HoleDefinition, distance_mm: Optional[float] = None) -> Dict[str, object]:
    data = hole.to_dict()
    if distance_mm is not None:
        data["distance_mm"] = float(distance_mm)
    return data


def _tip_to_dict(tip: TipPoint, calibrated: bool) -> Dict[str, object]:
    return {
        "label": tip.label,
        "conf": float(tip.conf),
        "track_id": tip.track_id,
        "frame_id": int(tip.frame_id),
        "px": float(tip.px),
        "py": float(tip.py),
        "center_px": [float(tip.px), float(tip.py)],
        "mm_x": float(tip.mm_x) if calibrated else None,
        "mm_y": float(tip.mm_y) if calibrated else None,
        "center_mm": [float(tip.mm_x), float(tip.mm_y)] if calibrated else None,
    }


def _serialize_step_state(step_state) -> Dict[str, object]:
    if step_state is None:
        return {
            "current_step_index": -1,
            "round_result": "unknown",
            "round_id": 0,
            "action_ng_step": -1,
            "action_ng_reason": "",
            "steps": [],
        }
    steps = []
    for step in getattr(step_state, "steps", []) or []:
        steps.append({
            "index": getattr(step, "index", -1),
            "class_name": getattr(step, "class_name", ""),
            "configured": bool(getattr(step, "configured", False)),
            "status": _enum_value(getattr(step, "status", "")),
            "enter_count": int(getattr(step, "enter_count", 0) or 0),
            "wrong_order_count": int(getattr(step, "wrong_order_count", 0) or 0),
            "ng_reason": getattr(step, "ng_reason", ""),
        })
    return {
        "current_step_index": _get_current_step_index(step_state),
        "round_result": _enum_value(getattr(step_state, "round_result", "unknown")),
        "round_id": int(getattr(step_state, "round_id", 0) or 0),
        "round_started_at": float(getattr(step_state, "round_started_at", 0.0) or 0.0),
        "round_completed_at": float(getattr(step_state, "round_completed_at", 0.0) or 0.0),
        "action_ng_step": int(getattr(step_state, "action_ng_step", -1) or -1),
        "action_ng_reason": getattr(step_state, "action_ng_reason", ""),
        "steps": steps,
    }


def _config_summary(config, calibrated: bool, holes: List[HoleDefinition]) -> Dict[str, object]:
    return {
        "tool_class_name": getattr(config, "tool_class_name", "扭力枪"),
        "hole_count": len(holes),
        "enabled_hole_count": len([h for h in holes if h.enabled]),
        "calibration_point_count": len(getattr(config, "calibration_points", []) or []),
        "calibration_valid": calibrated,
        "yolo_conf_threshold": float(getattr(config, "yolo_conf_threshold", 0.0) or 0.0),
        "yolo_iou_threshold": float(getattr(config, "yolo_iou_threshold", 0.0) or 0.0),
        "action_overlap_threshold": float(getattr(config, "action_overlap_threshold", 0.20) or 0.20),
        "action_pass_stable_frames": int(getattr(config, "action_pass_stable_frames", 2) or 2),
        "action_ng_stable_frames": int(getattr(config, "action_ng_stable_frames", 2) or 2),
        "action_order_constraint_enabled": bool(getattr(config, "action_order_constraint_enabled", True)),
        "point_judgement_enabled": bool(getattr(config, "point_judgement_enabled", False)),
    }


def _get_current_step_index(step_state) -> int:
    try:
        return int(getattr(step_state, "current_step_index", -1))
    except Exception:
        return -1


def _normalize_image_size(image_size: Optional[Iterable[int]]) -> Optional[List[int]]:
    if image_size is None:
        return None
    values = list(image_size)
    if len(values) < 2:
        return None
    return [int(values[0]), int(values[1])]


def _enum_value(value) -> str:
    return str(getattr(value, "value", value))


def _unique(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _inc_dict(data: Dict[str, int], key: str) -> None:
    data[key] = int(data.get(key, 0)) + 1


def _hole_key(hole_data: Dict[str, object]) -> str:
    if not hole_data:
        return ""
    name = str(hole_data.get("name", "") or "")
    step_index = hole_data.get("step_index", "")
    return name or f"step_{step_index}"


def _finalize_label_stats(stats: Dict[str, object]) -> Dict[str, object]:
    confs = [float(v) for v in stats.get("_confs", [])]
    return {
        "frame_count": int(stats.get("frame_count", 0)),
        "total_count": int(stats.get("total_count", 0)),
        "avg_conf": sum(confs) / len(confs) if confs else 0.0,
        "min_conf": min(confs) if confs else 0.0,
        "max_conf": max(confs) if confs else 0.0,
    }


def _finalize_overlap_stats(stats: Dict[str, object]) -> Dict[str, object]:
    polygon_ious = [float(v) for v in stats.get("_polygon_ious", [])]
    pair_frames = int(stats.get("pair_frames", 0) or 0)
    overlap_frames = int(stats.get("overlap_frames", 0) or 0)
    strong_frames = int(stats.get("strong_overlap_frames", 0) or 0)
    return {
        "pair_frames": pair_frames,
        "overlap_frames": overlap_frames,
        "medium_overlap_frames": int(stats.get("medium_overlap_frames", 0) or 0),
        "strong_overlap_frames": strong_frames,
        "overlap_rate": overlap_frames / pair_frames if pair_frames else 0.0,
        "strong_overlap_rate": strong_frames / pair_frames if pair_frames else 0.0,
        "max_polygon_iou": float(stats.get("max_polygon_iou", 0.0) or 0.0),
        "avg_polygon_iou": sum(polygon_ious) / len(polygon_ious) if polygon_ious else 0.0,
        "p95_polygon_iou": _percentile(sorted(polygon_ious), 95) if polygon_ious else 0.0,
        "max_bbox_iou": float(stats.get("max_bbox_iou", 0.0) or 0.0),
        "min_center_distance_px": stats.get("min_center_distance_px"),
        "longest_consecutive_overlap": int(stats.get("longest_consecutive_overlap", 0) or 0),
    }


def _distance_stats(values: Iterable[float]) -> Dict[str, float]:
    data = sorted(float(v) for v in values)
    if not data:
        return {}
    return {
        "min": data[0],
        "avg": sum(data) / len(data),
        "max": data[-1],
        "p50": _percentile(data, 50),
        "p95": _percentile(data, 95),
    }


def _percentile(sorted_values: List[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
