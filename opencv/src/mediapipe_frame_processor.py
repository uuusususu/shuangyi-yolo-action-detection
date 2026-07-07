"""MediaPipe gesture processor adapting runtime results for the Qt shell."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from config import ConfigManager
from mediapipe_gesture_runtime import (
    HAND_CONNECTIONS,
    NONE_LABEL,
    UNKNOWN_LABEL,
    GestureFrameResult,
    GestureStabilizerConfig,
    MediaPipeGestureRuntime,
    MediaPipeThresholds,
)
from models import DetectionOverlayState


TASK_TYPE = "mediapipe_gesture"


def _normalize_label(value: Any) -> str:
    return str(value or "").strip()


def resolve_mediapipe_task_path(config: ConfigManager, model_path: str | None = None) -> Path:
    raw_path = str(model_path or config.get_model_path() or "").strip()
    if not raw_path:
        raise FileNotFoundError("未配置 MediaPipe task 模型路径。")

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"MediaPipe task 模型文件不存在: {resolved}")

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
            project_root / "mediapipe_quick_task" / "exports" / path.name,
            project_root / "mediapipe_quick_task" / "exports" / "gesture_recognizer.task",
        ]
    )

    seen: set[str] = set()
    attempted: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        attempted.append(resolved)
        if resolved.exists():
            return resolved

    attempted_paths = ", ".join(str(candidate) for candidate in attempted)
    raise FileNotFoundError(f"MediaPipe task 模型文件不存在: {raw_path}。已尝试: {attempted_paths}")


RuntimeFactory = Callable[..., MediaPipeGestureRuntime]


class MediaPipeGestureProcessor:
    BOX_COLOR = (255, 180, 0)
    KEYPOINT_COLOR = (0, 255, 0)
    TEXT_COLOR = (255, 255, 255)
    ERROR_COLOR = (0, 0, 255)

    def __init__(self, config: ConfigManager, runtime_factory: RuntimeFactory | None = None) -> None:
        self.config = config
        self._runtime_factory = runtime_factory
        self.runtime: MediaPipeGestureRuntime | Any | None = None
        self.model: Any | None = None
        self._resolved_model_path = ""
        self._last_init_error = ""
        self._last_inference_snapshot: dict[str, Any] = {}
        self._last_overlay_state = DetectionOverlayState(task_type=TASK_TYPE)
        self._initialize()

    def _thresholds_from_config(self) -> MediaPipeThresholds:
        return MediaPipeThresholds(
            min_hand_detection_confidence=float(
                getattr(self.config, "mediapipe_min_hand_detection_confidence", 0.5)
            ),
            min_hand_presence_confidence=float(
                getattr(self.config, "mediapipe_min_hand_presence_confidence", 0.5)
            ),
            min_tracking_confidence=float(getattr(self.config, "mediapipe_min_tracking_confidence", 0.5)),
        )

    def _stabilizer_config_from_config(self) -> GestureStabilizerConfig:
        labels = tuple(
            label
            for label in [_normalize_label(item) for item in getattr(self.config, "category_names", [])]
            if label
        )
        if NONE_LABEL not in labels:
            labels = tuple(list(labels) + [NONE_LABEL])
        return GestureStabilizerConfig(
            score_threshold=float(getattr(self.config, "mediapipe_score_threshold", 0.65)),
            window_size=int(getattr(self.config, "gesture_window_size", 5)),
            enter_frames=int(getattr(self.config, "gesture_enter_frames", 3)),
            exit_frames=int(getattr(self.config, "gesture_exit_frames", 5)),
            labels=labels or ("fang", "wo", NONE_LABEL),
        )

    def _create_runtime(self, task_path: Path):
        if self._runtime_factory is not None:
            return self._runtime_factory(task_path, self.config)
        return MediaPipeGestureRuntime(
            task_path,
            num_hands=int(getattr(self.config, "mediapipe_num_hands", 2)),
            thresholds=self._thresholds_from_config(),
            stabilizer_config=self._stabilizer_config_from_config(),
        )

    def _initialize(self) -> None:
        self.runtime = None
        self.model = None
        self._last_init_error = ""
        try:
            task_path = resolve_mediapipe_task_path(self.config)
            runtime = self._create_runtime(task_path)
            runtime.load()
            self.runtime = runtime
            self.model = getattr(runtime, "recognizer", runtime)
            self._resolved_model_path = str(task_path)
            print(f"MediaPipe task 加载成功: {task_path}")
        except Exception as exc:
            self.runtime = None
            self.model = None
            self._resolved_model_path = str(getattr(self.config, "model_path", "") or "")
            self._last_init_error = f"初始化 MediaPipe 手势模型失败: {exc}"
            print(self._last_init_error)

    def reinitialize_if_needed(self) -> None:
        if self.runtime is None:
            self._initialize()

    def release(self) -> None:
        if self.runtime is not None:
            release = getattr(self.runtime, "release", None)
            if callable(release):
                release()
        self.runtime = None
        self.model = None

    def get_runtime_model_task(self) -> str:
        return TASK_TYPE

    def get_runtime_model_path(self) -> str:
        return self._resolved_model_path or self.config.get_model_path()

    def get_last_inference_snapshot(self) -> dict[str, Any]:
        return dict(self._last_inference_snapshot)

    def get_last_overlay_state(self) -> DetectionOverlayState:
        return DetectionOverlayState(
            source_frame_id=int(self._last_overlay_state.source_frame_id),
            timestamp=float(self._last_overlay_state.timestamp),
            model_path=str(self._last_overlay_state.model_path),
            task_type=str(self._last_overlay_state.task_type),
            detections=[dict(det) for det in self._last_overlay_state.detections],
            classification=dict(self._last_overlay_state.classification),
            matched_category_names=list(self._last_overlay_state.matched_category_names),
            actions=list(self._last_overlay_state.actions),
            status=str(self._last_overlay_state.status),
            error=str(self._last_overlay_state.error),
            round_id=int(self._last_overlay_state.round_id),
            official_speed_ms=dict(self._last_overlay_state.official_speed_ms),
        )

    def _configured_category_names(self) -> list[str]:
        names: list[str] = []
        for raw_name in getattr(self.config, "category_names", []):
            name = _normalize_label(raw_name)
            if name and name not in names:
                names.append(name)
        return names

    def _matched_names(self, result: GestureFrameResult) -> list[str]:
        stable_label = _normalize_label(result.stable_label)
        if stable_label in {"", UNKNOWN_LABEL, NONE_LABEL}:
            return []
        return [stable_label] if stable_label in self._configured_category_names() else []

    def _detections_from_result(self, result: GestureFrameResult, frame_shape) -> list[dict[str, Any]]:
        img_h, img_w = frame_shape[:2]
        detections: list[dict[str, Any]] = []
        for index, bbox in enumerate(result.bboxes):
            landmarks = result.hand_landmarks[index] if index < len(result.hand_landmarks) else []
            keypoints = []
            for item in landmarks:
                keypoints.append(
                    [
                        float(item.get("x", 0.0)) * float(img_w),
                        float(item.get("y", 0.0)) * float(img_h),
                    ]
                )
            detections.append(
                {
                    "class_id": index,
                    "label": str(result.stable_label or result.raw_label),
                    "raw_label": str(result.raw_label),
                    "stable_label": str(result.stable_label),
                    "conf": float(result.score),
                    "action": None,
                    "box": [float(bbox.left), float(bbox.top), float(bbox.right), float(bbox.bottom)],
                    "polygon": None,
                    "center": [float(bbox.center_x), float(bbox.center_y)],
                    "task_type": TASK_TYPE,
                    "handedness": result.handedness[index] if index < len(result.handedness) else "",
                    "keypoints": {
                        "points": keypoints,
                        "conf": [1.0 for _ in keypoints],
                        "connections": list(HAND_CONNECTIONS),
                    },
                    "landmarks": [dict(point) for point in landmarks],
                }
            )
        return detections

    def _unavailable_overlay(self, source_frame_id: int, round_id: int) -> DetectionOverlayState:
        timestamp = time.time()
        snapshot = {
            "timestamp": timestamp,
            "status": "session_unavailable",
            "task_type": TASK_TYPE,
            "model_path": self.get_runtime_model_path(),
            "config_path": str(self.config.get_config_path() or ""),
            "error": self._last_init_error,
            "source_frame_id": int(source_frame_id),
            "round_id": int(round_id),
            "matched_category_names": [],
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
            task_type=TASK_TYPE,
            status="session_unavailable",
            error=str(self._last_init_error),
            round_id=int(round_id),
        )
        return self.get_last_overlay_state()

    def process_overlay(
        self,
        frame: np.ndarray,
        *,
        source_frame_id: int = 0,
        round_id: int = 0,
    ) -> DetectionOverlayState:
        self.reinitialize_if_needed()
        if self.runtime is None:
            return self._unavailable_overlay(source_frame_id, round_id)

        try:
            result = self.runtime.recognize_frame(frame)
        except Exception as exc:
            timestamp = time.time()
            self._last_inference_snapshot = {
                "timestamp": timestamp,
                "status": "inference_error",
                "task_type": TASK_TYPE,
                "model_path": self.get_runtime_model_path(),
                "config_path": str(self.config.get_config_path() or ""),
                "error": str(exc),
                "source_frame_id": int(source_frame_id),
                "round_id": int(round_id),
                "matched_category_names": [],
            }
            self._last_overlay_state = DetectionOverlayState(
                source_frame_id=int(source_frame_id),
                timestamp=timestamp,
                model_path=str(self.get_runtime_model_path()),
                task_type=TASK_TYPE,
                status="inference_error",
                error=str(exc),
                round_id=int(round_id),
            )
            return self.get_last_overlay_state()

        timestamp = time.time()
        detections = self._detections_from_result(result, frame.shape)
        matched_category_names = self._matched_names(result)
        thresholds = getattr(self.runtime, "thresholds", self._thresholds_from_config())
        stabilizer_config = getattr(self.runtime, "stabilizer_config", self._stabilizer_config_from_config())

        snapshot = {
            "timestamp": timestamp,
            "status": "ok",
            "task_type": TASK_TYPE,
            "model_path": self.get_runtime_model_path(),
            "config_path": str(self.config.get_config_path() or ""),
            "source_frame_id": int(source_frame_id),
            "round_id": int(round_id),
            "raw_label": str(result.raw_label),
            "stable_label": str(result.stable_label),
            "score": float(result.score),
            "hand_count": int(result.hand_count),
            "handedness": list(result.handedness),
            "bboxes": [bbox.as_dict() for bbox in result.bboxes],
            "hand_landmarks": [[dict(point) for point in group] for group in result.hand_landmarks],
            "mediapipe_thresholds": thresholds.as_dict() if hasattr(thresholds, "as_dict") else {},
            "stabilizer": {
                "score_threshold": float(getattr(stabilizer_config, "score_threshold", 0.0)),
                "window_size": int(getattr(stabilizer_config, "window_size", 0)),
                "enter_frames": int(getattr(stabilizer_config, "enter_frames", 0)),
                "exit_frames": int(getattr(stabilizer_config, "exit_frames", 0)),
            },
            "inference_ms": float(result.inference_ms),
            "official_speed_ms": {"inference": float(result.inference_ms)},
            "tracking_diagnostics": dict(result.tracking_diagnostics),
            "detections": [dict(det) for det in detections],
            "classification": {},
            "actions": [],
            "matched_category_names": list(matched_category_names),
            "result_components": {
                "has_boxes": bool(result.bboxes),
                "has_masks": False,
                "has_keypoints": bool(result.hand_landmarks),
                "has_probs": False,
            },
        }
        self._last_inference_snapshot = snapshot
        self._last_overlay_state = DetectionOverlayState(
            source_frame_id=int(source_frame_id),
            timestamp=timestamp,
            model_path=str(self.get_runtime_model_path()),
            task_type=TASK_TYPE,
            detections=[dict(det) for det in detections],
            classification={},
            matched_category_names=list(matched_category_names),
            actions=[],
            status="ok",
            round_id=int(round_id),
            official_speed_ms={"inference": float(result.inference_ms)},
        )
        return self.get_last_overlay_state()

    def render_overlay(self, frame: np.ndarray, overlay_state: Optional[DetectionOverlayState]) -> np.ndarray:
        if overlay_state is None:
            return frame
        if str(getattr(overlay_state, "status", "")) in {"session_unavailable", "inference_error"}:
            cv2.putText(
                frame,
                str(getattr(overlay_state, "error", "") or "MediaPipe unavailable")[:96],
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.ERROR_COLOR,
                2,
            )
            return frame

        for det in list(getattr(overlay_state, "detections", []) or []):
            box = det.get("box") or [0, 0, 0, 0]
            if len(box) >= 4:
                x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.BOX_COLOR, 2)
                label = str(det.get("stable_label") or det.get("label") or "")
                conf = float(det.get("conf", 0.0))
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    self.TEXT_COLOR,
                    2,
                )
            keypoints = det.get("keypoints")
            if isinstance(keypoints, dict):
                self._draw_keypoints(frame, keypoints)
        return frame

    def _draw_keypoints(self, frame: np.ndarray, keypoints: dict[str, Any]) -> None:
        points = np.asarray(keypoints.get("points", []), dtype=np.float32)
        if points.size == 0:
            return
        if points.ndim == 1:
            points = points.reshape(-1, 2)
        for point in points:
            if len(point) >= 2:
                cv2.circle(frame, (int(point[0]), int(point[1])), 3, self.KEYPOINT_COLOR, -1)
        for start, end in list(keypoints.get("connections", []) or []):
            if start >= len(points) or end >= len(points):
                continue
            cv2.line(
                frame,
                (int(points[start][0]), int(points[start][1])),
                (int(points[end][0]), int(points[end][1])),
                self.KEYPOINT_COLOR,
                2,
            )
