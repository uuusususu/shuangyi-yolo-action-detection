"""MediaPipe gesture runtime for the desktop application."""

from __future__ import annotations

import shutil
import tempfile
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


UNKNOWN_LABEL = "unknown"
NONE_LABEL = "none"
DEFAULT_LABELS = ("fang", "wo", "none")
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


def sanitize_label(value: Any) -> str:
    if value is None:
        return UNKNOWN_LABEL
    label = str(value).replace("\r", "").replace("\n", "").strip()
    return label or UNKNOWN_LABEL


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _contains_non_ascii(value: Path) -> bool:
    try:
        str(value).encode("ascii")
    except UnicodeEncodeError:
        return True
    return False


def _landmark_value(item: Any, key: str) -> float:
    if isinstance(item, dict):
        return _safe_float(item.get(key, 0.0))
    return _safe_float(getattr(item, key, 0.0))


@dataclass(frozen=True)
class HandBBox:
    left: int
    top: int
    right: int
    bottom: int
    center_x: int
    center_y: int

    @property
    def center(self) -> tuple[int, int]:
        return self.center_x, self.center_y

    @property
    def area(self) -> int:
        return max(0, self.right - self.left) * max(0, self.bottom - self.top)

    def as_dict(self) -> dict[str, int]:
        return {
            "left": int(self.left),
            "top": int(self.top),
            "right": int(self.right),
            "bottom": int(self.bottom),
            "center_x": int(self.center_x),
            "center_y": int(self.center_y),
        }


@dataclass(frozen=True)
class MediaPipeThresholds:
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    def as_dict(self) -> dict[str, float]:
        return {
            "min_hand_detection_confidence": float(self.min_hand_detection_confidence),
            "min_hand_presence_confidence": float(self.min_hand_presence_confidence),
            "min_tracking_confidence": float(self.min_tracking_confidence),
        }


@dataclass(frozen=True)
class GestureStabilizerConfig:
    score_threshold: float = 0.65
    window_size: int = 5
    enter_frames: int = 3
    exit_frames: int = 5
    labels: tuple[str, ...] = DEFAULT_LABELS
    unknown_label: str = UNKNOWN_LABEL


class GestureStabilizer:
    def __init__(self, config: GestureStabilizerConfig | None = None) -> None:
        self.config = config or GestureStabilizerConfig()
        self._window: deque[str] = deque(maxlen=max(1, int(self.config.window_size)))
        self._stable_label = self.config.unknown_label
        self._candidate_label = self.config.unknown_label
        self._candidate_count = 0
        self._exit_count = 0

    @property
    def stable_label(self) -> str:
        return self._stable_label

    def reset(self) -> None:
        self._window.clear()
        self._stable_label = self.config.unknown_label
        self._candidate_label = self.config.unknown_label
        self._candidate_count = 0
        self._exit_count = 0

    def _normalize_prediction(self, label: Any, score: float) -> str:
        normalized = sanitize_label(label)
        if _safe_float(score) < float(self.config.score_threshold):
            return self.config.unknown_label
        if normalized not in set(self.config.labels):
            return self.config.unknown_label
        return normalized

    def _window_candidate(self) -> str:
        if not self._window:
            return self.config.unknown_label
        counts = Counter(self._window)
        max_count = max(counts.values())
        tied = {label for label, count in counts.items() if count == max_count}
        for label in reversed(self._window):
            if label in tied:
                return label
        return self.config.unknown_label

    def _track_candidate(self, candidate: str) -> None:
        if candidate == self._candidate_label:
            self._candidate_count += 1
        else:
            self._candidate_label = candidate
            self._candidate_count = 1

    def update(self, label: Any, score: float) -> str:
        normalized = self._normalize_prediction(label, score)
        self._window.append(normalized)
        candidate = self._window_candidate()
        exit_threshold = max(1, int(self.config.exit_frames))
        enter_threshold = max(1, int(self.config.enter_frames))

        if candidate == self._stable_label:
            if self._stable_label != self.config.unknown_label and normalized != self._stable_label:
                self._exit_count += 1
                if self._exit_count >= exit_threshold:
                    self._stable_label = self.config.unknown_label
                    self._exit_count = 0
                    self._track_candidate(self.config.unknown_label)
                    return self._stable_label
            else:
                self._exit_count = 0
            self._track_candidate(candidate)
            return self._stable_label

        if candidate == self.config.unknown_label:
            self._track_candidate(candidate)
            if self._stable_label != self.config.unknown_label:
                self._exit_count += 1
            if self._exit_count >= exit_threshold:
                self._stable_label = self.config.unknown_label
                self._exit_count = 0
            return self._stable_label

        self._track_candidate(candidate)
        if self._stable_label == self.config.unknown_label:
            if self._candidate_count >= enter_threshold:
                self._stable_label = candidate
                self._exit_count = 0
            return self._stable_label

        if normalized != self._stable_label:
            self._exit_count += 1
        else:
            self._exit_count = 0
        if self._candidate_count >= enter_threshold and self._exit_count >= exit_threshold:
            self._stable_label = candidate
            self._exit_count = 0
        elif self._exit_count >= exit_threshold:
            self._stable_label = self.config.unknown_label
            self._exit_count = 0
        return self._stable_label


@dataclass
class GestureFrameResult:
    raw_label: str = UNKNOWN_LABEL
    score: float = 0.0
    stable_label: str = UNKNOWN_LABEL
    handedness: list[str] = field(default_factory=list)
    hand_landmarks: list[list[dict[str, float]]] = field(default_factory=list)
    bboxes: list[HandBBox] = field(default_factory=list)
    inference_ms: float = 0.0
    timestamp_ms: int = 0
    tracking_diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def hand_count(self) -> int:
        return max(len(self.hand_landmarks), len(self.bboxes), len(self.handedness))

    def to_log_fields(self) -> dict[str, Any]:
        return {
            "raw_label": self.raw_label,
            "stable_label": self.stable_label,
            "score": float(self.score),
            "hand_count": int(self.hand_count),
            "handedness": list(self.handedness),
            "bboxes": [bbox.as_dict() for bbox in self.bboxes],
            "hand_landmarks": [[dict(point) for point in group] for group in self.hand_landmarks],
            "inference_ms": float(self.inference_ms),
            "timestamp_ms": int(self.timestamp_ms),
            "tracking_diagnostics": dict(self.tracking_diagnostics),
        }


def normalize_landmarks(landmarks: Sequence[Any]) -> list[dict[str, float]]:
    normalized: list[dict[str, float]] = []
    for item in landmarks:
        normalized.append(
            {
                "x": min(max(_landmark_value(item, "x"), 0.0), 1.0),
                "y": min(max(_landmark_value(item, "y"), 0.0), 1.0),
                "z": _landmark_value(item, "z"),
            }
        )
    return normalized


def compute_hand_bbox(
    landmarks: Sequence[Any],
    *,
    frame_width: int,
    frame_height: int,
    padding_px: int = 0,
) -> HandBBox:
    width = max(1, int(frame_width))
    height = max(1, int(frame_height))
    xs = [min(max(_landmark_value(item, "x"), 0.0), 1.0) for item in landmarks]
    ys = [min(max(_landmark_value(item, "y"), 0.0), 1.0) for item in landmarks]
    if not xs or not ys:
        return HandBBox(0, 0, 0, 0, 0, 0)

    left = max(0, int(round(min(xs) * width)) - int(padding_px))
    right = min(width - 1, int(round(max(xs) * width)) + int(padding_px))
    top = max(0, int(round(min(ys) * height)) - int(padding_px))
    bottom = min(height - 1, int(round(max(ys) * height)) + int(padding_px))
    center_x = int(round((left + right) / 2))
    center_y = int(round((top + bottom) / 2))
    return HandBBox(left=left, top=top, right=right, bottom=bottom, center_x=center_x, center_y=center_y)


def _category_label(category: Any) -> str:
    for attr in ("category_name", "display_name", "label"):
        value = getattr(category, attr, None)
        if value:
            return sanitize_label(value)
    return UNKNOWN_LABEL


def _category_score(category: Any) -> float:
    return _safe_float(getattr(category, "score", 0.0))


def extract_gesture_frame_result(
    result: Any,
    *,
    frame_width: int,
    frame_height: int,
) -> GestureFrameResult:
    raw_label = UNKNOWN_LABEL
    raw_score = 0.0
    for gesture_group in getattr(result, "gestures", []) or []:
        if not gesture_group:
            continue
        category = gesture_group[0]
        score = _category_score(category)
        if score >= raw_score:
            raw_label = _category_label(category)
            raw_score = score

    handedness: list[str] = []
    for hand_group in getattr(result, "handedness", []) or []:
        if hand_group:
            handedness.append(_category_label(hand_group[0]))

    landmark_groups: list[list[dict[str, float]]] = []
    bboxes: list[HandBBox] = []
    for landmark_group in getattr(result, "hand_landmarks", []) or []:
        normalized = normalize_landmarks(landmark_group)
        landmark_groups.append(normalized)
        if normalized:
            bboxes.append(compute_hand_bbox(normalized, frame_width=frame_width, frame_height=frame_height))

    return GestureFrameResult(
        raw_label=raw_label,
        score=float(raw_score),
        handedness=handedness,
        hand_landmarks=landmark_groups,
        bboxes=bboxes,
    )


def bgr_to_mediapipe_image(bgr_frame: Any):
    import cv2  # type: ignore
    import mediapipe as mp  # type: ignore

    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


def create_mediapipe_recognizer(
    task_path: Path,
    *,
    num_hands: int,
    thresholds: MediaPipeThresholds,
):
    from mediapipe.tasks import python  # type: ignore
    from mediapipe.tasks.python import vision  # type: ignore

    options = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=str(task_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=int(num_hands),
        min_hand_detection_confidence=float(thresholds.min_hand_detection_confidence),
        min_hand_presence_confidence=float(thresholds.min_hand_presence_confidence),
        min_tracking_confidence=float(thresholds.min_tracking_confidence),
    )
    return vision.GestureRecognizer.create_from_options(options)


RecognizerFactory = Callable[[Path, int, MediaPipeThresholds], Any]
ImageConverter = Callable[[Any], Any]


class MediaPipeGestureRuntime:
    def __init__(
        self,
        task_path: str | Path,
        *,
        num_hands: int = 2,
        thresholds: MediaPipeThresholds | None = None,
        stabilizer_config: GestureStabilizerConfig | None = None,
        recognizer_factory: RecognizerFactory | None = None,
        image_converter: ImageConverter | None = None,
    ) -> None:
        self.task_path = Path(task_path)
        self.num_hands = max(1, int(num_hands))
        self.thresholds = thresholds or MediaPipeThresholds()
        self.stabilizer_config = stabilizer_config or GestureStabilizerConfig()
        self.stabilizer = GestureStabilizer(self.stabilizer_config)
        self._recognizer_factory = recognizer_factory or (
            lambda path, num_hands, thresholds: create_mediapipe_recognizer(
                path,
                num_hands=num_hands,
                thresholds=thresholds,
            )
        )
        self._image_converter = image_converter or bgr_to_mediapipe_image
        self._recognizer: Any | None = None
        self._prepared_task_path: Path | None = None
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._started_at = time.perf_counter()
        self._last_timestamp_ms = -1

    @property
    def recognizer(self) -> Any | None:
        return self._recognizer

    @property
    def prepared_task_path(self) -> Path | None:
        return self._prepared_task_path

    def _prepare_task_path(self) -> Path:
        resolved_task = Path(self.task_path).expanduser().resolve()
        if not resolved_task.exists():
            raise FileNotFoundError(f"MediaPipe task file does not exist: {resolved_task}")

        if not _contains_non_ascii(resolved_task):
            self._prepared_task_path = resolved_task
            return resolved_task

        self._temp_dir = tempfile.TemporaryDirectory(prefix="mp_task_", ignore_cleanup_errors=True)
        prepared = Path(self._temp_dir.name) / resolved_task.name
        shutil.copy2(resolved_task, prepared)
        self._prepared_task_path = prepared
        return prepared

    def load(self) -> None:
        if self._recognizer is not None:
            return
        prepared = self._prepare_task_path()
        try:
            self._recognizer = self._recognizer_factory(prepared, self.num_hands, self.thresholds)
        except Exception:
            if self._temp_dir is not None:
                try:
                    self._temp_dir.cleanup()
                except PermissionError:
                    pass
                self._temp_dir = None
            self._prepared_task_path = None
            raise

    def _next_timestamp_ms(self) -> int:
        timestamp_ms = int((time.perf_counter() - self._started_at) * 1000.0)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def recognize_frame(self, bgr_frame: Any) -> GestureFrameResult:
        self.load()
        if self._recognizer is None:
            raise RuntimeError("MediaPipe recognizer is not loaded.")
        if bgr_frame is None or not hasattr(bgr_frame, "shape"):
            raise ValueError("A BGR numpy frame is required.")

        height, width = bgr_frame.shape[:2]
        timestamp_ms = self._next_timestamp_ms()
        started_at = time.perf_counter()
        image = self._image_converter(bgr_frame)
        raw_result = self._recognizer.recognize_for_video(image, int(timestamp_ms))
        inference_ms = (time.perf_counter() - started_at) * 1000.0

        result = extract_gesture_frame_result(raw_result, frame_width=width, frame_height=height)
        result.stable_label = self.stabilizer.update(result.raw_label, result.score)
        result.inference_ms = float(inference_ms)
        result.timestamp_ms = int(timestamp_ms)
        result.tracking_diagnostics = self._tracking_diagnostics(result)
        return result

    def _tracking_diagnostics(self, result: GestureFrameResult) -> dict[str, Any]:
        if not result.bboxes:
            return {
                "bbox_center_x": None,
                "bbox_center_y": None,
                "bbox_area": None,
            }
        bbox = result.bboxes[0]
        return {
            "bbox_center_x": int(bbox.center_x),
            "bbox_center_y": int(bbox.center_y),
            "bbox_area": int(bbox.area),
        }

    def release(self) -> None:
        if self._recognizer is not None:
            close = getattr(self._recognizer, "close", None)
            if callable(close):
                close()
        self._recognizer = None
        self.stabilizer.reset()
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except PermissionError:
                pass
            self._temp_dir = None
        self._prepared_task_path = None
        self._last_timestamp_ms = -1
        self._started_at = time.perf_counter()
