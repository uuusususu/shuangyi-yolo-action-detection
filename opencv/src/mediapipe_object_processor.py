"""MediaPipe object detector processor and multi-detector composition."""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import cv2
import numpy as np

from config import ConfigManager, DEFAULT_OBJECT_MODEL_PATH
from models import DetectionOverlayState


OBJECT_TASK_TYPE = "mediapipe_object_detection"
GESTURE_TASK_TYPE = "mediapipe_gesture"
COMPOSITE_TASK_TYPE = "mediapipe_object_detection+mediapipe_gesture"
UNKNOWN_LABEL = "unknown"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _contains_non_ascii(value: Path) -> bool:
    try:
        str(value).encode("ascii")
    except UnicodeEncodeError:
        return True
    return False


def _sanitize_label(value: Any) -> str:
    label = str(value or "").replace("\r", "").replace("\n", "").strip()
    return label or UNKNOWN_LABEL


@dataclass(frozen=True)
class ObjectBBox:
    left: int
    top: int
    right: int
    bottom: int
    width: int
    height: int
    center_x: int
    center_y: int

    @property
    def center(self) -> tuple[int, int]:
        return self.center_x, self.center_y

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def clipped(self, *, frame_width: int, frame_height: int) -> "ObjectBBox":
        width = max(1, int(frame_width))
        height = max(1, int(frame_height))
        left = min(max(0, self.left), width - 1)
        top = min(max(0, self.top), height - 1)
        right = min(max(left, self.right), width - 1)
        bottom = min(max(top, self.bottom), height - 1)
        box_width = max(0, right - left)
        box_height = max(0, bottom - top)
        return ObjectBBox(
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            width=box_width,
            height=box_height,
            center_x=int(round((left + right) / 2)),
            center_y=int(round((top + bottom) / 2)),
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "left": int(self.left),
            "top": int(self.top),
            "right": int(self.right),
            "bottom": int(self.bottom),
            "width": int(self.width),
            "height": int(self.height),
            "center_x": int(self.center_x),
            "center_y": int(self.center_y),
            "area": int(self.area),
        }


@dataclass(frozen=True)
class ObjectDetection:
    label: str
    score: float
    bbox: ObjectBBox

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": str(self.label),
            "score": float(self.score),
            "bbox": self.bbox.as_dict(),
        }


@dataclass(frozen=True)
class ObjectFrameResult:
    timestamp_ms: int
    inference_ms: float
    detections: tuple[ObjectDetection, ...]

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    @property
    def top_label(self) -> str:
        if not self.detections:
            return UNKNOWN_LABEL
        return str(self.detections[0].label)

    @property
    def top_score(self) -> float:
        if not self.detections:
            return 0.0
        return float(self.detections[0].score)

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ms": int(self.timestamp_ms),
            "inference_ms": float(self.inference_ms),
            "detection_count": int(self.detection_count),
            "top_label": self.top_label,
            "top_score": float(self.top_score),
            "detections": [item.as_dict() for item in self.detections],
        }


class ObjectDetectorHandle:
    def __init__(self, detector: Any, temp_dir: tempfile.TemporaryDirectory[str] | None = None) -> None:
        self._detector = detector
        self._temp_dir = temp_dir

    def __getattr__(self, name: str) -> Any:
        return getattr(self._detector, name)

    def close(self) -> None:
        close = getattr(self._detector, "close", None)
        if callable(close):
            close()
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except PermissionError:
                pass
            self._temp_dir = None


class _ImageObjectDetector:
    def __init__(self, detector: Any, handle: ObjectDetectorHandle) -> None:
        self.detector = detector
        self._handle = handle

    def detect_bgr(self, bgr_frame: np.ndarray) -> ObjectFrameResult:
        mp_image = bgr_to_mediapipe_image(bgr_frame)
        started = time.perf_counter()
        result = self.detector.detect(mp_image)
        inference_ms = (time.perf_counter() - started) * 1000.0
        return extract_object_frame_result(result, timestamp_ms=0, inference_ms=inference_ms)

    def close(self) -> None:
        self._handle.close()


def _resolve_model_for_mediapipe(model_path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    resolved_model = Path(model_path).resolve()
    if not resolved_model.exists():
        raise FileNotFoundError(f"MediaPipe object 模型文件不存在: {resolved_model}")
    if not _contains_non_ascii(resolved_model):
        return resolved_model, None
    temp_dir = tempfile.TemporaryDirectory(prefix="mp_object_model_", ignore_cleanup_errors=True)
    temp_model_path = Path(temp_dir.name) / resolved_model.name
    shutil.copy2(resolved_model, temp_model_path)
    return temp_model_path, temp_dir


def resolve_mediapipe_object_model_path(config: ConfigManager, model_path: str | None = None) -> Path:
    raw_path = str(model_path or getattr(config, "object_model_path", "") or "").strip()
    if not raw_path:
        raw_path = DEFAULT_OBJECT_MODEL_PATH
    if not raw_path.lower().endswith(".tflite"):
        raise FileNotFoundError(f"MediaPipe object 模型仅支持 .tflite 文件: {raw_path}")

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"MediaPipe object 模型文件不存在: {resolved}")

    config_path = getattr(config, "_config_path", None)
    opencv_dir = Path(config_path).resolve().parent if config_path else Path(__file__).resolve().parents[1]
    project_root = opencv_dir.parent
    candidates = []
    if config_path:
        candidates.append(Path(config_path).resolve().parent / path)
    candidates.extend(
        [
            Path.cwd() / path,
            opencv_dir / path,
            project_root / path,
            project_root / "mediapipe-samples-main" / "model" / "training_runs" / "exported_model" / path.name,
        ]
    )

    attempted: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        attempted.append(resolved)
        if resolved.exists():
            return resolved
    attempted_text = ", ".join(str(item) for item in attempted)
    raise FileNotFoundError(f"MediaPipe object 模型文件不存在: {raw_path}。已尝试: {attempted_text}")


def bgr_to_mediapipe_image(bgr_frame: np.ndarray):
    import mediapipe as mp  # type: ignore

    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


def _bbox_from_mediapipe(value: Any) -> ObjectBBox:
    left = _safe_int(getattr(value, "origin_x", 0))
    top = _safe_int(getattr(value, "origin_y", 0))
    width = max(0, _safe_int(getattr(value, "width", 0)))
    height = max(0, _safe_int(getattr(value, "height", 0)))
    right = left + width
    bottom = top + height
    return ObjectBBox(
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        width=width,
        height=height,
        center_x=int(round(left + width / 2)),
        center_y=int(round(top + height / 2)),
    )


def _category_label(category: Any) -> str:
    for attr in ("category_name", "display_name", "label"):
        value = getattr(category, attr, None)
        if value:
            return _sanitize_label(value)
    return UNKNOWN_LABEL


def extract_object_frame_result(result: Any, *, timestamp_ms: int, inference_ms: float = 0.0) -> ObjectFrameResult:
    detections: list[ObjectDetection] = []
    for detection in getattr(result, "detections", []) or []:
        categories = getattr(detection, "categories", []) or []
        if categories:
            category = categories[0]
            label = _category_label(category)
            score = _safe_float(getattr(category, "score", 0.0))
        else:
            label = UNKNOWN_LABEL
            score = 0.0
        bbox = _bbox_from_mediapipe(getattr(detection, "bounding_box", None))
        detections.append(ObjectDetection(label=label, score=score, bbox=bbox))
    detections.sort(key=lambda item: item.score, reverse=True)
    return ObjectFrameResult(
        timestamp_ms=int(timestamp_ms),
        inference_ms=float(inference_ms),
        detections=tuple(detections),
    )


def create_object_detector(model_path: Path, *, score_threshold: float, max_results: int) -> _ImageObjectDetector:
    from mediapipe.tasks import python  # type: ignore
    from mediapipe.tasks.python import vision  # type: ignore

    model, temp_dir = _resolve_model_for_mediapipe(Path(model_path))
    try:
        options = vision.ObjectDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.IMAGE,
            score_threshold=float(score_threshold),
            max_results=int(max_results),
        )
        detector = vision.ObjectDetector.create_from_options(options)
        handle = ObjectDetectorHandle(detector, temp_dir)
        return _ImageObjectDetector(detector, handle)
    except Exception:
        if temp_dir is not None:
            temp_dir.cleanup()
        raise


ObjectDetectorFactory = Callable[[Path, ConfigManager], Any]


class MediaPipeObjectDetectorProcessor:
    BOX_COLOR = (255, 0, 0)
    CENTER_COLOR = (0, 255, 255)
    TEXT_COLOR = (255, 255, 255)
    ERROR_COLOR = (0, 0, 255)

    def __init__(self, config: ConfigManager, detector_factory: ObjectDetectorFactory | None = None) -> None:
        self.config = config
        self._detector_factory = detector_factory
        self.detector: Any | None = None
        self.model: Any | None = None
        self._resolved_model_path = ""
        self._last_init_error = ""
        self._last_inference_snapshot: dict[str, Any] = {}
        self._last_overlay_state = DetectionOverlayState(task_type=OBJECT_TASK_TYPE)
        self._initialize()

    def _create_detector(self, model_path: Path) -> Any:
        if self._detector_factory is not None:
            return self._detector_factory(model_path, self.config)
        return create_object_detector(
            model_path,
            score_threshold=float(getattr(self.config, "object_score_threshold", 0.3)),
            max_results=int(getattr(self.config, "object_max_results", 5)),
        )

    def _initialize(self) -> None:
        self.detector = None
        self.model = None
        self._last_init_error = ""
        try:
            model_path = resolve_mediapipe_object_model_path(self.config)
            detector = self._create_detector(model_path)
            self.detector = detector
            self.model = getattr(detector, "detector", detector)
            self._resolved_model_path = str(model_path)
            print(f"MediaPipe object 模型加载成功: {model_path}")
        except Exception as exc:
            self.detector = None
            self.model = None
            self._resolved_model_path = str(getattr(self.config, "object_model_path", "") or "")
            self._last_init_error = f"初始化 MediaPipe 目标检测模型失败: {exc}"
            print(self._last_init_error)

    def reinitialize_if_needed(self) -> None:
        if self.detector is None:
            self._initialize()

    def release(self) -> None:
        if self.detector is not None:
            close = getattr(self.detector, "close", None)
            if callable(close):
                close()
        self.detector = None
        self.model = None

    def get_runtime_model_task(self) -> str:
        return OBJECT_TASK_TYPE

    def get_runtime_model_path(self) -> str:
        return self._resolved_model_path or str(getattr(self.config, "object_model_path", "") or "")

    def get_last_inference_snapshot(self) -> dict[str, Any]:
        return dict(self._last_inference_snapshot)

    def _unavailable_overlay(self, source_frame_id: int, round_id: int) -> DetectionOverlayState:
        timestamp = time.time()
        snapshot = {
            "timestamp": timestamp,
            "status": "session_unavailable",
            "task_type": OBJECT_TASK_TYPE,
            "model_path": self.get_runtime_model_path(),
            "object_model_path": self.get_runtime_model_path(),
            "config_path": str(self.config.get_config_path() or ""),
            "error": self._last_init_error,
            "source_frame_id": int(source_frame_id),
            "round_id": int(round_id),
            "matched_category_names": [],
            "object_detection": {"detections": []},
            "result_components": {
                "has_boxes": False,
                "has_masks": False,
                "has_keypoints": False,
                "has_probs": False,
            },
        }
        self._last_inference_snapshot = snapshot
        self._last_overlay_state = DetectionOverlayState(
            source_frame_id=int(source_frame_id),
            timestamp=timestamp,
            model_path=str(self.get_runtime_model_path()),
            task_type=OBJECT_TASK_TYPE,
            status="session_unavailable",
            error=str(self._last_init_error),
            round_id=int(round_id),
        )
        return self._copy_overlay_state(self._last_overlay_state)

    @staticmethod
    def _copy_overlay_state(overlay_state: DetectionOverlayState) -> DetectionOverlayState:
        return DetectionOverlayState(
            source_frame_id=int(overlay_state.source_frame_id),
            timestamp=float(overlay_state.timestamp),
            model_path=str(overlay_state.model_path),
            task_type=str(overlay_state.task_type),
            detections=[dict(det) for det in overlay_state.detections],
            classification=dict(overlay_state.classification),
            matched_category_names=list(overlay_state.matched_category_names),
            actions=list(overlay_state.actions),
            status=str(overlay_state.status),
            error=str(overlay_state.error),
            round_id=int(overlay_state.round_id),
            official_speed_ms=dict(overlay_state.official_speed_ms),
        )

    def _detections_from_result(self, result: ObjectFrameResult, frame_shape) -> list[dict[str, Any]]:
        height, width = frame_shape[:2]
        detections: list[dict[str, Any]] = []
        for index, detection in enumerate(result.detections):
            bbox = detection.bbox.clipped(frame_width=width, frame_height=height)
            detections.append(
                {
                    "class_id": index,
                    "label": str(detection.label),
                    "conf": float(detection.score),
                    "action": None,
                    "box": [float(bbox.left), float(bbox.top), float(bbox.right), float(bbox.bottom)],
                    "polygon": None,
                    "center": [float(bbox.center_x), float(bbox.center_y)],
                    "task_type": OBJECT_TASK_TYPE,
                    "source": "object",
                    "object_bbox": bbox.as_dict(),
                }
            )
        return detections

    def process_overlay(
        self,
        frame: np.ndarray,
        *,
        source_frame_id: int = 0,
        round_id: int = 0,
    ) -> DetectionOverlayState:
        self.reinitialize_if_needed()
        if self.detector is None:
            return self._unavailable_overlay(source_frame_id, round_id)

        try:
            detect_bgr = getattr(self.detector, "detect_bgr", None)
            if callable(detect_bgr):
                result = detect_bgr(frame)
            else:
                started = time.perf_counter()
                raw_result = self.detector.detect(bgr_to_mediapipe_image(frame))
                result = extract_object_frame_result(
                    raw_result,
                    timestamp_ms=0,
                    inference_ms=(time.perf_counter() - started) * 1000.0,
                )
        except Exception as exc:
            timestamp = time.time()
            self._last_inference_snapshot = {
                "timestamp": timestamp,
                "status": "inference_error",
                "task_type": OBJECT_TASK_TYPE,
                "model_path": self.get_runtime_model_path(),
                "object_model_path": self.get_runtime_model_path(),
                "config_path": str(self.config.get_config_path() or ""),
                "error": str(exc),
                "source_frame_id": int(source_frame_id),
                "round_id": int(round_id),
                "matched_category_names": [],
                "object_detection": {"detections": []},
            }
            self._last_overlay_state = DetectionOverlayState(
                source_frame_id=int(source_frame_id),
                timestamp=timestamp,
                model_path=str(self.get_runtime_model_path()),
                task_type=OBJECT_TASK_TYPE,
                status="inference_error",
                error=str(exc),
                round_id=int(round_id),
            )
            return self._copy_overlay_state(self._last_overlay_state)

        timestamp = time.time()
        detections = self._detections_from_result(result, frame.shape)
        object_payload = result.as_dict()
        object_payload["detections"] = [dict(item) for item in detections]
        snapshot = {
            "timestamp": timestamp,
            "status": "ok",
            "task_type": OBJECT_TASK_TYPE,
            "model_path": self.get_runtime_model_path(),
            "object_model_path": self.get_runtime_model_path(),
            "config_path": str(self.config.get_config_path() or ""),
            "source_frame_id": int(source_frame_id),
            "round_id": int(round_id),
            "object_score_threshold": float(getattr(self.config, "object_score_threshold", 0.3)),
            "object_max_results": int(getattr(self.config, "object_max_results", 5)),
            "object_result_hold_ms": int(getattr(self.config, "object_result_hold_ms", 250)),
            "top_label": result.top_label,
            "top_score": float(result.top_score),
            "object_detection": object_payload,
            "detections": [dict(det) for det in detections],
            "classification": {},
            "actions": [],
            "matched_category_names": [],
            "official_speed_ms": {"object_detection": float(result.inference_ms)},
            "result_components": {
                "has_boxes": bool(detections),
                "has_masks": False,
                "has_keypoints": False,
                "has_probs": False,
            },
        }
        self._last_inference_snapshot = snapshot
        self._last_overlay_state = DetectionOverlayState(
            source_frame_id=int(source_frame_id),
            timestamp=timestamp,
            model_path=str(self.get_runtime_model_path()),
            task_type=OBJECT_TASK_TYPE,
            detections=[dict(det) for det in detections],
            classification={},
            matched_category_names=[],
            actions=[],
            status="ok",
            round_id=int(round_id),
            official_speed_ms={"object_detection": float(result.inference_ms)},
        )
        return self._copy_overlay_state(self._last_overlay_state)

    def render_overlay(self, frame: np.ndarray, overlay_state: Optional[DetectionOverlayState]) -> np.ndarray:
        if overlay_state is None:
            return frame
        status = str(getattr(overlay_state, "status", "") or "")
        if status in {"session_unavailable", "inference_error"}:
            cv2.putText(
                frame,
                str(getattr(overlay_state, "error", "") or "Object detector unavailable")[:96],
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.ERROR_COLOR,
                2,
            )
            return frame

        for det in list(getattr(overlay_state, "detections", []) or []):
            if str(det.get("source", "")) != "object" and str(det.get("task_type", "")) != OBJECT_TASK_TYPE:
                continue
            box = det.get("box") or [0, 0, 0, 0]
            if len(box) < 4:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.BOX_COLOR, 3)
            center = det.get("center") or []
            if len(center) >= 2:
                cv2.circle(frame, (int(float(center[0])), int(float(center[1]))), 4, self.CENTER_COLOR, -1)
            label = f"{det.get('label', UNKNOWN_LABEL)} {float(det.get('conf', 0.0)):.2f}"
            text_x = max(0, x1 + 8)
            text_y = max(18, y1 + 18)
            cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 1.2, self.TEXT_COLOR, 1, cv2.LINE_AA)
        return frame


class CompositeFrameProcessor:
    def __init__(self, processors: Sequence[Any], *, mode_display_name: str = "") -> None:
        self.processors = list(processors)
        self.mode_display_name = mode_display_name or "目标检测 + 手势检测"
        self._last_overlay_state = DetectionOverlayState(task_type=COMPOSITE_TASK_TYPE)
        self._last_child_states: list[DetectionOverlayState] = []
        self._last_inference_snapshot: dict[str, Any] = {}

    def release(self) -> None:
        for processor in self.processors:
            release = getattr(processor, "release", None)
            if callable(release):
                release()

    def get_runtime_model_task(self) -> str:
        return "+".join(self._processor_task(processor) for processor in self.processors)

    def get_runtime_model_path(self) -> str:
        return " | ".join(self._processor_path(processor) for processor in self.processors if self._processor_path(processor))

    def get_last_inference_snapshot(self) -> dict[str, Any]:
        return dict(self._last_inference_snapshot)

    @staticmethod
    def _processor_task(processor: Any) -> str:
        getter = getattr(processor, "get_runtime_model_task", None)
        if callable(getter):
            return str(getter())
        return str(getattr(processor, "task_type", "unknown"))

    @staticmethod
    def _processor_path(processor: Any) -> str:
        getter = getattr(processor, "get_runtime_model_path", None)
        if callable(getter):
            return str(getter())
        return ""

    @staticmethod
    def _mode_name_from_task(task_type: str) -> str:
        if task_type == OBJECT_TASK_TYPE:
            return "object"
        if task_type == GESTURE_TASK_TYPE:
            return "gesture"
        return str(task_type or "unknown")

    def _child_snapshot(self, processor: Any) -> dict[str, Any]:
        getter = getattr(processor, "get_last_inference_snapshot", None)
        if callable(getter):
            result = getter()
            if isinstance(result, dict):
                return dict(result)
        return {}

    def process_overlay(
        self,
        frame: np.ndarray,
        *,
        source_frame_id: int = 0,
        round_id: int = 0,
    ) -> DetectionOverlayState:
        started = time.perf_counter()
        child_states: list[DetectionOverlayState] = []
        detections: list[dict[str, Any]] = []
        matched: list[str] = []
        official_speed_ms: dict[str, float] = {}
        processor_snapshots: dict[str, dict[str, Any]] = {}
        inference_order: list[str] = []
        model_paths: list[str] = []
        status = "ok"
        error_parts: list[str] = []

        for processor in self.processors:
            task_type = self._processor_task(processor)
            mode_name = self._mode_name_from_task(task_type)
            inference_order.append(mode_name)
            overlay = processor.process_overlay(frame, source_frame_id=source_frame_id, round_id=round_id)
            child_states.append(overlay)
            detections.extend(dict(det) for det in getattr(overlay, "detections", []) or [])
            if task_type != OBJECT_TASK_TYPE:
                for name in list(getattr(overlay, "matched_category_names", []) or []):
                    if name and name not in matched:
                        matched.append(str(name))
            official_speed_ms.update(dict(getattr(overlay, "official_speed_ms", {}) or {}))
            child_snapshot = self._child_snapshot(processor)
            processor_snapshots[mode_name] = child_snapshot
            model_path = str(getattr(overlay, "model_path", "") or "")
            if model_path:
                model_paths.append(model_path)
            child_status = str(getattr(overlay, "status", "") or "")
            if child_status and child_status != "ok":
                status = child_status
                child_error = str(getattr(overlay, "error", "") or "")
                if child_error:
                    error_parts.append(child_error)

        timestamp = time.time()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        official_speed_ms.setdefault("total", float(elapsed_ms))
        result_components = {
            "has_boxes": any(bool(det.get("box")) for det in detections),
            "has_masks": False,
            "has_keypoints": any(bool(det.get("keypoints") or det.get("landmarks")) for det in detections),
            "has_probs": False,
        }
        enabled_modes = list(inference_order)
        self._last_child_states = child_states
        self._last_overlay_state = DetectionOverlayState(
            source_frame_id=int(source_frame_id),
            timestamp=timestamp,
            model_path=" | ".join(model_paths),
            task_type=self.get_runtime_model_task(),
            detections=detections,
            classification={},
            matched_category_names=matched,
            actions=[],
            status=status,
            error="; ".join(error_parts),
            round_id=int(round_id),
            official_speed_ms=official_speed_ms,
        )
        self._last_inference_snapshot = {
            "timestamp": timestamp,
            "status": status,
            "task_type": self.get_runtime_model_task(),
            "model_path": self._last_overlay_state.model_path,
            "source_frame_id": int(source_frame_id),
            "round_id": int(round_id),
            "enabled_detection_modes": enabled_modes,
            "inference_order": inference_order,
            "processors": processor_snapshots,
            "object_detection": processor_snapshots.get("object", {}).get("object_detection", {"detections": []}),
            "gesture_detection": processor_snapshots.get("gesture", {}),
            "detections": [dict(det) for det in detections],
            "matched_category_names": list(matched),
            "official_speed_ms": official_speed_ms,
            "result_components": result_components,
        }
        return DetectionOverlayState(
            source_frame_id=self._last_overlay_state.source_frame_id,
            timestamp=self._last_overlay_state.timestamp,
            model_path=self._last_overlay_state.model_path,
            task_type=self._last_overlay_state.task_type,
            detections=[dict(det) for det in detections],
            classification={},
            matched_category_names=list(matched),
            actions=[],
            status=status,
            error=self._last_overlay_state.error,
            round_id=int(round_id),
            official_speed_ms=dict(official_speed_ms),
        )

    def render_overlay(self, frame: np.ndarray, overlay_state: Optional[DetectionOverlayState]) -> np.ndarray:
        rendered = frame
        child_states = list(self._last_child_states)
        if not child_states and overlay_state is not None:
            child_states = [overlay_state for _ in self.processors]
        for processor, child_state in zip(self.processors, child_states):
            render = getattr(processor, "render_overlay", None)
            if callable(render):
                rendered = render(rendered, child_state)
        return rendered


def create_frame_processor_from_config(
    config: ConfigManager,
    *,
    gesture_processor_cls: Any | None = None,
    object_processor_cls: Any | None = None,
    object_detector_factory: ObjectDetectorFactory | None = None,
) -> Any:
    if not config.has_enabled_detection_mode():
        raise ValueError("至少启用一种检测模式。")

    processors: list[Any] = []
    if bool(getattr(config, "enable_object_detection", False)):
        object_cls = object_processor_cls or MediaPipeObjectDetectorProcessor
        processors.append(object_cls(config, detector_factory=object_detector_factory))
    if bool(getattr(config, "enable_gesture_detection", True)):
        if gesture_processor_cls is None:
            from mediapipe_frame_processor import MediaPipeGestureProcessor

            gesture_processor_cls = MediaPipeGestureProcessor
        processors.append(gesture_processor_cls(config))

    if len(processors) == 1:
        return processors[0]
    return CompositeFrameProcessor(processors, mode_display_name=config.get_detection_mode_display_name())

