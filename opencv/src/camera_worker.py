"""Camera worker module."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage

from config import ConfigManager
from detection_logger import DetectionSessionLogger
from gesture_trigger_cycle import DebouncedGestureState
from models import DetectionOverlayState, PipelineStats, PreviewFrame, RoundCompleted, RoundProgress
from mvsdk_camera import MvSdkCamera, MvSdkCapture
from state import AppState


class CameraWorker(QObject):
    """Coordinate camera capture, inference scheduling, and runtime events."""

    frame_ready = Signal(QImage)
    error_occurred = Signal(str)
    state_changed = Signal(str, bool)
    fps_updated = Signal(float)
    gesture_state_changed = Signal(str, bool)
    round_progress_changed = Signal(dict)
    round_completed = Signal(dict)
    pipeline_stats_updated = Signal(dict)

    def __init__(self, config: ConfigManager, state: AppState):
        super().__init__()
        self.config = config
        self.state = state

        self.cap: Optional[object] = None
        self.running = False
        self.frame_processor = None

        self._mvsdk_camera: Optional[MvSdkCamera] = None
        self._open_error_message = ""

        self._capture_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None

        self._preview_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._display_lock = threading.Lock()
        self._overlay_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._latest_lock = self._preview_lock

        self._latest_preview_frame: Optional[PreviewFrame] = None
        self._latest_infer_frame: Optional[PreviewFrame] = None
        self._latest_display_frame: Optional[PreviewFrame] = None
        self._latest_display_qimage: Optional[QImage] = None
        self._latest_overlay_state = DetectionOverlayState()
        self._latest_frame: Optional[np.ndarray] = None

        self._next_frame_id = 0
        self._last_inferred_frame_id = 0
        self._capture_fail_count = 0
        self._reconnect_count = 0
        self._reconnect_fail_threshold = 30
        self._max_reconnect_attempts = 3

        self._capture_count = 0
        self._capture_fps_start_time = time.time()
        self._preview_count = 0
        self._preview_fps_start_time = time.time()
        self._infer_count = 0
        self._infer_fps_start_time = time.time()
        self._pipeline_stats = PipelineStats()
        self._last_logged_display_strategy: Optional[str] = None

        self._category_states: Dict[str, DebouncedGestureState] = {}
        self._category_state_signature: Tuple[str, ...] = tuple()
        self._refresh_category_states(emit_reset=False)

        self._inference_frame_index = 0
        self._detection_logger: Optional[DetectionSessionLogger] = None
        self._detection_log_path: Optional[Path] = None

        self._round_id = 1
        self._seen_targets: Set[str] = set()
        self._round_started_at = time.time()
        self._round_hold_until = 0.0
        self._round_holding = False

    def set_frame_processor(self, processor) -> None:
        self.frame_processor = processor
        runtime_task_getter = getattr(processor, "get_runtime_model_task", None)
        runtime_path_getter = getattr(processor, "get_runtime_model_path", None)
        runtime_task = (
            str(runtime_task_getter()).strip().lower()
            if callable(runtime_task_getter)
            else self.config.get_model_task()
        )
        runtime_path = (
            str(runtime_path_getter()).strip()
            if callable(runtime_path_getter)
            else self.config.get_model_path()
        )
        print(
            f"[runtime] frame_processor_set config={self.config.get_config_path()} "
            f"model={runtime_path} task={runtime_task}"
        )

    @staticmethod
    def _normalize_category_name(name: object) -> str:
        if name is None:
            return ""
        return str(name).strip()

    def _active_category_names(self) -> List[str]:
        names: List[str] = []
        for raw_name in getattr(self.config, "category_names", []):
            normalized = self._normalize_category_name(raw_name)
            if normalized and normalized not in names:
                names.append(normalized)
        return names

    def _refresh_category_states(self, *, emit_reset: bool = True) -> None:
        names = tuple(self._active_category_names())
        if names == self._category_state_signature:
            return

        if emit_reset:
            self._reset_category_states(emit=True)

        self._category_states = {
            name: DebouncedGestureState(
                name=name,
                n_on=self.config.static_n_on,
                n_off=self.config.static_n_off,
            )
            for name in names
        }
        self._category_state_signature = names

    def _reset_category_states(self, *, emit: bool) -> None:
        emitted_names: Set[str] = set()
        for state in self._category_states.values():
            events = state.reset()
            if not emit:
                continue
            for name, active in events:
                emitted_names.add(str(name))
                self.gesture_state_changed.emit(name, active)
        if emit:
            for name in self._active_category_names():
                if name in emitted_names:
                    continue
                self.gesture_state_changed.emit(name, False)

    def _copy_overlay_state(self, overlay_state: DetectionOverlayState) -> DetectionOverlayState:
        return DetectionOverlayState(
            source_frame_id=int(overlay_state.source_frame_id),
            timestamp=float(overlay_state.timestamp),
            model_path=str(overlay_state.model_path),
            task_type=str(getattr(overlay_state, "task_type", "")),
            detections=[dict(det) for det in overlay_state.detections],
            classification=dict(getattr(overlay_state, "classification", {}) or {}),
            matched_category_names=list(overlay_state.matched_category_names),
            actions=list(overlay_state.actions),
            status=str(overlay_state.status),
            error=str(overlay_state.error),
            round_id=int(overlay_state.round_id),
            official_speed_ms=dict(getattr(overlay_state, "official_speed_ms", {}) or {}),
        )

    def get_latest_overlay_state(self) -> DetectionOverlayState:
        with self._overlay_lock:
            return self._copy_overlay_state(self._latest_overlay_state)

    def get_latest_preview_qimage(self) -> Optional[QImage]:
        preview_frame: Optional[PreviewFrame] = None
        with self._preview_lock:
            if self._latest_preview_frame is not None:
                preview_frame = PreviewFrame(
                    frame_id=int(self._latest_preview_frame.frame_id),
                    timestamp=float(self._latest_preview_frame.timestamp),
                    image=self._latest_preview_frame.image.copy(),
                    width=int(self._latest_preview_frame.width),
                    height=int(self._latest_preview_frame.height),
                )

        if preview_frame is None:
            return None

        self._update_preview_stats(preview_frame)
        return self._frame_to_qimage(preview_frame.image)

    def _get_display_overlay_state(self) -> Optional[DetectionOverlayState]:
        use_overlay = bool(getattr(self.config, "display_sync_to_inference", True))
        max_age_ms = float(getattr(self.config, "display_result_max_age_ms", 250.0))
        self._pipeline_stats.display_strategy = "preview_only"
        self._pipeline_stats.display_result_age_ms = 0.0
        self._pipeline_stats.display_source_frame_id = 0

        if not self.state.inference_on or not use_overlay:
            return None

        overlay_state = self.get_latest_overlay_state()
        status = str(getattr(overlay_state, "status", "idle") or "idle")
        if status in {"session_unavailable", "inference_error"}:
            self._pipeline_stats.display_strategy = "preview_with_overlay"
            self._pipeline_stats.display_source_frame_id = int(overlay_state.source_frame_id)
            self._pipeline_stats.display_result_age_ms = max(
                0.0, (time.time() - float(overlay_state.timestamp)) * 1000.0
            )
            return overlay_state

        if status != "ok" or float(getattr(overlay_state, "timestamp", 0.0)) <= 0.0:
            return None

        age_ms = max(0.0, (time.time() - float(overlay_state.timestamp)) * 1000.0)
        self._pipeline_stats.display_result_age_ms = age_ms
        self._pipeline_stats.display_source_frame_id = int(overlay_state.source_frame_id)
        if age_ms <= max_age_ms:
            self._pipeline_stats.display_strategy = "preview_with_overlay"
            return overlay_state

        self._pipeline_stats.display_strategy = "preview_with_stale_overlay_hidden"
        return None

    def get_latest_display_payload(self) -> tuple[Optional[QImage], Optional[DetectionOverlayState]]:
        if self.state.inference_on:
            with self._display_lock:
                display_frame = self._latest_display_frame
                display_qimage = self._latest_display_qimage
            if display_frame is not None and display_qimage is not None:
                self._pipeline_stats.display_strategy = "official_rendered_result"
                self._pipeline_stats.display_result_age_ms = max(
                    0.0, (time.time() - float(display_frame.timestamp)) * 1000.0
                )
                self._pipeline_stats.display_source_frame_id = int(display_frame.frame_id)
                self._log_display_strategy_if_changed()
                self._emit_pipeline_stats()
                return display_qimage.copy(), None

        qimage = self.get_latest_preview_qimage()
        self._pipeline_stats.display_strategy = (
            "raw_preview_waiting_result" if self.state.inference_on else "raw_preview"
        )
        self._pipeline_stats.display_result_age_ms = 0.0
        self._pipeline_stats.display_source_frame_id = 0
        self._log_display_strategy_if_changed()
        self._emit_pipeline_stats()
        return qimage, None

    def get_latest_display_qimage(self) -> Optional[QImage]:
        """Backward-compatible helper returning the latest preview frame."""
        qimage, _overlay_state = self.get_latest_display_payload()
        return qimage

    def get_latest_qimage(self) -> Optional[QImage]:
        return self.get_latest_preview_qimage()

    def _resolve_runtime_base_dir(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def _start_detection_logging(self) -> None:
        self._inference_frame_index = 0
        self._detection_log_path = None
        self._detection_logger = None

        if not bool(getattr(self.config, "detection_log_enabled", True)):
            return

        try:
            self._detection_logger = DetectionSessionLogger(
                base_dir=self._resolve_runtime_base_dir(),
                log_dir=str(getattr(self.config, "detection_log_dir", "logs/detections")),
                flush_interval=int(getattr(self.config, "detection_log_flush_interval", 30)),
            )
            enabled_modes = list(getattr(self.config, "get_enabled_detection_modes", lambda: ["gesture"])())
            metadata = {
                "config_path": str(self.config.get_config_path() or ""),
                "runtime": str(getattr(self.config, "get_detection_mode_task_type", lambda: "mediapipe_gesture")()),
                "enabled_detection_modes": enabled_modes,
                "inference_order": enabled_modes,
                "model_path": str(self.config.get_model_path()),
                "mediapipe_task_path": str(getattr(self.config, "mediapipe_task_path", "")),
                "object_model_path": str(getattr(self.config, "object_model_path", "")),
                "task_type": str(self.config.get_model_task()),
                "mediapipe_thresholds": {
                    "score_threshold": float(getattr(self.config, "mediapipe_score_threshold", 0.0)),
                    "min_hand_detection_confidence": float(
                        getattr(self.config, "mediapipe_min_hand_detection_confidence", 0.0)
                    ),
                    "min_hand_presence_confidence": float(
                        getattr(self.config, "mediapipe_min_hand_presence_confidence", 0.0)
                    ),
                    "min_tracking_confidence": float(
                        getattr(self.config, "mediapipe_min_tracking_confidence", 0.0)
                    ),
                },
                "gesture_stabilizer": {
                    "window_size": int(getattr(self.config, "gesture_window_size", 0)),
                    "enter_frames": int(getattr(self.config, "gesture_enter_frames", 0)),
                    "exit_frames": int(getattr(self.config, "gesture_exit_frames", 0)),
                },
                "category_names": list(getattr(self.config, "category_names", [])),
            }
            self._detection_log_path = self._detection_logger.start_session(metadata)
        except Exception as exc:
            self._detection_logger = None
            self._detection_log_path = None
            self.error_occurred.emit(f"妫€娴嬫棩蹇楀垵濮嬪寲澶辫触: {exc}")

    def _stop_detection_logging(self, reason: str) -> None:
        if self._detection_logger is None:
            return
        try:
            self._detection_logger.stop_session(
                {
                    "reason": reason,
                    "total_frames_seen": int(self._inference_frame_index),
                }
            )
        except Exception:
            pass
        finally:
            self._detection_logger = None
            self._detection_log_path = None
            self._inference_frame_index = 0

    def _build_detection_log_record(
        self,
        *,
        overlay_state: DetectionOverlayState,
        snapshot: Optional[dict],
        frame_shape: tuple[int, ...],
    ) -> dict:
        snapshot = snapshot or {}
        return {
            "frame_index": int(self._inference_frame_index),
            "source_frame_id": int(overlay_state.source_frame_id),
            "round_id": int(overlay_state.round_id),
            "status": str(getattr(overlay_state, "status", "")),
            "task_type": str(getattr(overlay_state, "task_type", "")),
            "model_path": str(getattr(overlay_state, "model_path", "")),
            "raw_label": snapshot.get("raw_label"),
            "stable_label": snapshot.get("stable_label"),
            "score": snapshot.get("score"),
            "hand_count": snapshot.get("hand_count"),
            "bboxes": snapshot.get("bboxes", []),
            "hand_landmarks": snapshot.get("hand_landmarks", []),
            "matched_category_names": list(getattr(overlay_state, "matched_category_names", []) or []),
            "detections": [dict(det) for det in getattr(overlay_state, "detections", []) or []],
            "tracking_diagnostics": dict(snapshot.get("tracking_diagnostics", {}) or {}),
            "enabled_detection_modes": list(snapshot.get("enabled_detection_modes", []) or []),
            "inference_order": list(snapshot.get("inference_order", []) or []),
            "object_detection": dict(snapshot.get("object_detection", {}) or {}),
            "gesture_detection": dict(snapshot.get("gesture_detection", {}) or {}),
            "frame_shape": [int(v) for v in frame_shape],
            "pipeline_stats": self._pipeline_stats.to_dict(),
            "snapshot": snapshot,
        }

    def _log_detection_frame_if_needed(
        self,
        *,
        overlay_state: DetectionOverlayState,
        frame_shape: tuple[int, ...],
    ) -> None:
        if self._detection_logger is None:
            return

        every_n = max(1, int(getattr(self.config, "detection_log_every_n_frames", 1)))
        if ((self._inference_frame_index - 1) % every_n) != 0:
            return

        snapshot = {}
        if self.frame_processor is not None:
            getter = getattr(self.frame_processor, "get_last_inference_snapshot", None)
            if callable(getter):
                result = getter()
                if isinstance(result, dict):
                    snapshot = result

        record = self._build_detection_log_record(
            overlay_state=overlay_state,
            snapshot=snapshot,
            frame_shape=frame_shape,
        )
        self._detection_logger.log_frame(record)

    def _current_round_progress(self) -> RoundProgress:
        target_names = self._active_category_names()
        return RoundProgress(
            round_id=int(self._round_id),
            seen_targets=sorted(self._seen_targets),
            target_count=len(target_names),
            completed_count=len(self._seen_targets),
            holding=bool(self._round_holding),
        )

    def _emit_round_progress(self) -> None:
        self.round_progress_changed.emit(self._current_round_progress().to_dict())

    def _reset_round_state(self, *, emit_progress: bool, new_round: bool) -> None:
        if new_round:
            self._round_id += 1
        self._seen_targets.clear()
        self._round_started_at = time.time()
        self._round_hold_until = 0.0
        self._round_holding = False
        if emit_progress:
            self._emit_round_progress()

    def _mark_round_completed(self) -> None:
        completed_at = time.time()
        payload = RoundCompleted(
            round_id=int(self._round_id),
            completed_targets=sorted(self._seen_targets),
            completed_at=completed_at,
            duration_ms=int(max(0.0, (completed_at - self._round_started_at) * 1000.0)),
        )
        self._round_holding = True
        self._round_hold_until = completed_at + float(getattr(self.config, "round_cooldown_seconds", 2.0))
        self.round_completed.emit(payload.to_dict())
        self._emit_round_progress()

    def _update_round_state(self, matched_names: Set[str]) -> None:
        if self._round_holding:
            return

        before = set(self._seen_targets)
        self._seen_targets.update(matched_names)
        if before != self._seen_targets:
            self._emit_round_progress()

        target_names = self._active_category_names()
        if target_names and all(target in self._seen_targets for target in target_names):
            self._mark_round_completed()

    def _maybe_finish_round_hold(self) -> None:
        if not self._round_holding:
            return
        if time.time() < self._round_hold_until:
            return

        self._reset_category_states(emit=True)
        self._reset_round_state(emit_progress=True, new_round=True)

    def _update_category_states(self, matched_names: Set[str]) -> None:
        self._refresh_category_states(emit_reset=True)
        if self._round_holding:
            return
        for name, state in self._category_states.items():
            state.n_on = self.config.static_n_on
            state.n_off = self.config.static_n_off
            for event_name, active in state.update(name in matched_names):
                self.gesture_state_changed.emit(event_name, active)

    def _set_overlay_state(self, overlay_state: DetectionOverlayState) -> None:
        with self._overlay_lock:
            self._latest_overlay_state = self._copy_overlay_state(overlay_state)

    def _clear_overlay_state(self) -> None:
        self._set_overlay_state(DetectionOverlayState())

    def _clear_display_frame(self) -> None:
        with self._display_lock:
            self._latest_display_frame = None
            self._latest_display_qimage = None

    def _overlay_state_from_snapshot(self, snapshot: dict, *, source_frame_id: int) -> DetectionOverlayState:
        return DetectionOverlayState(
            source_frame_id=int(snapshot.get("source_frame_id", source_frame_id)),
            timestamp=float(snapshot.get("timestamp", time.time())),
            model_path=str(snapshot.get("model_path", "")),
            task_type=str(snapshot.get("task_type", "")),
            detections=[dict(det) for det in snapshot.get("detections", []) or []],
            classification=dict(snapshot.get("classification", {}) or {}),
            matched_category_names=list(snapshot.get("matched_category_names", []) or []),
            actions=list(snapshot.get("actions", []) or []),
            status=str(snapshot.get("status", "ok")),
            error=str(snapshot.get("error", "")),
            round_id=int(snapshot.get("round_id", self._round_id)),
        )

    def _emit_pipeline_stats(self) -> None:
        self.pipeline_stats_updated.emit(self._pipeline_stats.to_dict())

    def _log_display_strategy_if_changed(self) -> None:
        display_strategy = str(self._pipeline_stats.display_strategy or "")
        if not display_strategy or display_strategy == self._last_logged_display_strategy:
            return
        self._last_logged_display_strategy = display_strategy
        print(
            f"[runtime] display_strategy={display_strategy} task={self._pipeline_stats.task_type} "
            f"model={self._pipeline_stats.model_path} config={self._pipeline_stats.config_path} "
            f"boxes={self._pipeline_stats.has_boxes} masks={self._pipeline_stats.has_masks} "
            f"keypoints={self._pipeline_stats.has_keypoints} probs={self._pipeline_stats.has_probs} "
            f"results={self._pipeline_stats.result_count}"
        )

    def _update_capture_fps(self) -> None:
        self._capture_count += 1
        elapsed = time.time() - self._capture_fps_start_time
        if elapsed < 1.0:
            return

        self._pipeline_stats.capture_fps = self._capture_count / elapsed
        self._capture_count = 0
        self._capture_fps_start_time = time.time()
        self.fps_updated.emit(self._pipeline_stats.capture_fps)
        self.state.set_fps(self._pipeline_stats.capture_fps)
        self._emit_pipeline_stats()

    def _update_fps(self) -> None:
        """Backward-compatible wrapper for legacy tests."""
        self._update_capture_fps()

    def _update_preview_stats(self, preview_frame: PreviewFrame) -> None:
        self._preview_count += 1
        self._pipeline_stats.preview_latency_ms = max(
            0.0, (time.time() - float(preview_frame.timestamp)) * 1000.0
        )
        elapsed = time.time() - self._preview_fps_start_time
        if elapsed >= 1.0:
            self._pipeline_stats.preview_fps = self._preview_count / elapsed
            self._preview_count = 0
            self._preview_fps_start_time = time.time()
        self._emit_pipeline_stats()

    def _update_infer_stats(self, overlay_state: DetectionOverlayState, *, infer_elapsed_ms: float) -> None:
        self._infer_count += 1
        self._pipeline_stats.infer_latency_ms = max(0.0, float(infer_elapsed_ms))
        self._pipeline_stats.display_source_frame_id = int(overlay_state.source_frame_id)
        self._pipeline_stats.task_type = str(getattr(overlay_state, "task_type", "") or "")
        self._pipeline_stats.model_path = str(getattr(overlay_state, "model_path", "") or "")
        self._pipeline_stats.config_path = str(self.config.get_config_path() or "")
        snapshot = {}
        if self.frame_processor is not None:
            getter = getattr(self.frame_processor, "get_last_inference_snapshot", None)
            if callable(getter):
                result = getter()
                if isinstance(result, dict):
                    snapshot = result
        components = dict(snapshot.get("result_components", {}) or {})
        self._pipeline_stats.result_count = int(len(getattr(overlay_state, "detections", []) or []))
        self._pipeline_stats.has_boxes = bool(components.get("has_boxes", False) or components.get("has_obb", False))
        self._pipeline_stats.has_masks = bool(components.get("has_masks", False))
        self._pipeline_stats.has_keypoints = bool(components.get("has_keypoints", False))
        self._pipeline_stats.has_probs = bool(components.get("has_probs", False))
        self._pipeline_stats.processor_latency_ms = dict(
            snapshot.get("official_speed_ms", getattr(overlay_state, "official_speed_ms", {}) or {}) or {}
        )
        elapsed = time.time() - self._infer_fps_start_time
        if elapsed >= 1.0:
            self._pipeline_stats.infer_fps = self._infer_count / elapsed
            self._infer_count = 0
            self._infer_fps_start_time = time.time()
        self._emit_pipeline_stats()

    def _open_capture(self) -> bool:
        self._open_error_message = ""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self._mvsdk_camera is not None:
            try:
                self._mvsdk_camera.close()
            except Exception:
                pass
            self._mvsdk_camera = None

        if not MvSdkCamera.is_available():
            err = MvSdkCamera.load_error()
            if err:
                self._open_error_message = f"mvsdk 鍒濆鍖栧け璐? {err}"
            else:
                self._open_error_message = "mvsdk 涓嶅彲鐢紝璇锋鏌ラ┍鍔ㄦ垨 SDK 鏄惁瀹夎"
            return False

        friendly = str(getattr(self.config, "mvsdk_friendly_name", "") or "")
        if not friendly:
            devices = MvSdkCamera.enumerate_devices()
            if not devices:
                self._open_error_message = "鏈娴嬪埌杩堝痉濞佽鐩告満"
                return False
            friendly = str(devices[0].friendly_name)
            self.config.mvsdk_friendly_name = friendly
            self.config.save()

        cam = MvSdkCamera()
        if not cam.open(friendly_name=friendly):
            self._open_error_message = f"鏃犳硶鎵撳紑杩堝痉濞佽鐩告満: {friendly}"
            return False

        self._mvsdk_camera = cam
        self.cap = MvSdkCapture(cam)
        return True

    def _stop_camera_from_thread(self, error_message: str) -> None:
        self.running = False

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if self._mvsdk_camera is not None:
            try:
                self._mvsdk_camera.close()
            except Exception:
                pass
            self._mvsdk_camera = None

        with self._preview_lock:
            self._latest_preview_frame = None
            self._latest_frame = None
        with self._infer_lock:
            self._latest_infer_frame = None

        self._clear_overlay_state()
        self._clear_display_frame()
        self._stop_detection_logging("camera_stop_from_thread")
        self._reset_category_states(emit=True)
        self._reset_round_state(emit_progress=True, new_round=False)

        if self.state.inference_on:
            self.state.set_inference_on(False)
            self.state_changed.emit("inference_on", False)

        self.state.set_camera_on(False)
        self.state_changed.emit("camera_on", False)
        self.error_occurred.emit(error_message)

    @Slot()
    def start_camera(self) -> None:
        if self.cap is not None or self.running:
            return

        try:
            if not self._open_capture():
                msg = self._open_error_message or "无法打开迈德威视相机，请检查设备连接。"
                self.error_occurred.emit(msg)
                return

            self.running = True
            self.state.set_camera_on(True)
            self.state_changed.emit("camera_on", True)

            self._capture_count = 0
            self._capture_fps_start_time = time.time()
            self._preview_count = 0
            self._preview_fps_start_time = time.time()
            self._infer_count = 0
            self._infer_fps_start_time = time.time()
            self._pipeline_stats = PipelineStats()
            self._emit_pipeline_stats()

            self._capture_fail_count = 0
            self._reconnect_count = 0
            self._next_frame_id = 0
            self._last_inferred_frame_id = 0
            self._reset_round_state(emit_progress=True, new_round=False)

            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()

            self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self._inference_thread.start()
        except Exception as exc:
            self.error_occurred.emit(f"鍚姩鎽勫儚澶村け璐? {exc}")
            self.cap = None

    @Slot()
    def stop_camera(self) -> None:
        self.running = False

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=2.0)
            self._inference_thread = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if self._mvsdk_camera is not None:
            try:
                self._mvsdk_camera.close()
            except Exception:
                pass
            self._mvsdk_camera = None

        with self._preview_lock:
            self._latest_preview_frame = None
            self._latest_frame = None
        with self._infer_lock:
            self._latest_infer_frame = None

        self._clear_overlay_state()
        self._clear_display_frame()
        self._stop_detection_logging("camera_stop")
        self._reset_category_states(emit=True)
        self._reset_round_state(emit_progress=True, new_round=False)

        if self.state.inference_on:
            self.state.set_inference_on(False)
            self.state_changed.emit("inference_on", False)

        self.state.set_camera_on(False)
        self.state_changed.emit("camera_on", False)

    @Slot()
    def start_inference(self) -> None:
        if not self.state.camera_on:
            self.error_occurred.emit("请先开启相机。")
            return
        if not bool(getattr(self.config, "has_enabled_detection_mode", lambda: True)()):
            self.error_occurred.emit("至少启用一种检测模式。")
            return

        print(
            f"[runtime] start_inference config={self.config.get_config_path()} "
            f"model={self.config.get_model_path()} mode={self.config.get_detection_mode_display_name()}"
        )
        self.state.set_inference_on(True)
        self.state_changed.emit("inference_on", True)

        self._refresh_category_states(emit_reset=False)
        self._reset_category_states(emit=False)
        self._reset_round_state(emit_progress=True, new_round=False)
        self._clear_overlay_state()
        self._clear_display_frame()
        self._start_detection_logging()

    @Slot()
    def stop_inference(self) -> None:
        self._stop_detection_logging("manual_stop_inference")
        self.state.set_inference_on(False)
        self.state_changed.emit("inference_on", False)

        self._clear_overlay_state()
        self._clear_display_frame()
        self._reset_category_states(emit=True)
        self._reset_round_state(emit_progress=True, new_round=False)

    @Slot()
    def run_single_test(self) -> None:
        if not self.state.camera_on:
            self.error_occurred.emit("请先开启相机。")
            return
        if self.state.inference_on:
            self.error_occurred.emit("当前正在连续检测，请先停止后再执行测试。")
            return

        self._refresh_category_states(emit_reset=False)
        self._reset_category_states(emit=True)
        self._reset_round_state(emit_progress=True, new_round=False)
        self._clear_overlay_state()
        self._emit_pipeline_stats()

    def _capture_loop(self) -> None:
        reconnect_attempts = 0

        while self.running and self.cap is not None:
            self._maybe_finish_round_hold()

            ret, frame = self.cap.read()
            if not ret:
                self._capture_fail_count += 1
                if self._capture_fail_count >= int(self._reconnect_fail_threshold):
                    self._capture_fail_count = 0
                    if reconnect_attempts >= int(self._max_reconnect_attempts):
                        self._stop_camera_from_thread("读取摄像头帧失败，请检查设备。")
                        break

                    reconnect_attempts += 1
                    self._reconnect_count += 1
                    self.error_occurred.emit("相机采集失败，正在尝试重连...")
                    ok = self._open_capture()
                    if not ok:
                        time.sleep(0.2)
                    continue

                time.sleep(0.01)
                continue

            reconnect_attempts = 0
            self._capture_fail_count = 0

            if frame is None or frame.size == 0:
                time.sleep(0.01)
                continue

            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            frame_id = self._next_frame_id + 1
            self._next_frame_id = frame_id
            timestamp = time.time()
            preview_frame = PreviewFrame(
                frame_id=frame_id,
                timestamp=timestamp,
                image=frame.copy(),
                width=int(frame.shape[1]),
                height=int(frame.shape[0]),
            )

            with self._preview_lock:
                self._latest_preview_frame = preview_frame
                self._latest_frame = preview_frame.image.copy()
            with self._infer_lock:
                self._latest_infer_frame = preview_frame

            self._update_fps()
            time.sleep(0.001)

    def _inference_loop(self) -> None:
        while self.running:
            self._maybe_finish_round_hold()

            if not self.state.inference_on or self.frame_processor is None:
                time.sleep(0.01)
                continue

            preview_frame: Optional[PreviewFrame] = None
            with self._infer_lock:
                if self._latest_infer_frame is not None:
                    preview_frame = self._latest_infer_frame

            if preview_frame is None:
                time.sleep(0.005)
                continue

            frame_id = int(preview_frame.frame_id)
            if frame_id <= self._last_inferred_frame_id:
                time.sleep(0.005)
                continue

            if self._last_inferred_frame_id > 0 and frame_id > (self._last_inferred_frame_id + 1):
                self._pipeline_stats.dropped_for_infer += frame_id - self._last_inferred_frame_id - 1

            try:
                infer_started_at = time.time()
                self._inference_frame_index += 1
                if hasattr(self.frame_processor, "process_overlay"):
                    overlay_state = self.frame_processor.process_overlay(
                        preview_frame.image,
                        source_frame_id=frame_id,
                        round_id=self._round_id,
                    )
                else:
                    self.frame_processor.process(preview_frame.image.copy())
                    getter = getattr(self.frame_processor, "get_last_inference_snapshot", None)
                    snapshot = getter() if callable(getter) else {}
                    overlay_state = self._overlay_state_from_snapshot(
                        snapshot if isinstance(snapshot, dict) else {},
                        source_frame_id=frame_id,
                    )
                self._last_inferred_frame_id = frame_id
                self._set_overlay_state(overlay_state)
                if self.frame_processor is not None:
                    if hasattr(self.frame_processor, "render_official_result"):
                        rendered_frame = self.frame_processor.render_official_result(
                            preview_frame.image.copy()
                        )
                    elif hasattr(self.frame_processor, "render_overlay"):
                        rendered_frame = self.frame_processor.render_overlay(
                            preview_frame.image.copy(),
                            overlay_state,
                        )
                    else:
                        rendered_frame = preview_frame.image.copy()
                    with self._display_lock:
                        self._latest_display_frame = PreviewFrame(
                            frame_id=frame_id,
                            timestamp=float(overlay_state.timestamp),
                            image=rendered_frame,
                            width=int(rendered_frame.shape[1]),
                            height=int(rendered_frame.shape[0]),
                        )
                        self._latest_display_qimage = self._frame_to_qimage(rendered_frame)

                matched_names = {
                    self._normalize_category_name(name)
                    for name in overlay_state.matched_category_names
                    if self._normalize_category_name(name)
                }
                self._update_category_states(matched_names)
                self._update_round_state(matched_names)
                self._update_infer_stats(
                    overlay_state,
                    infer_elapsed_ms=(time.time() - infer_started_at) * 1000.0,
                )
                self._log_detection_frame_if_needed(
                    overlay_state=overlay_state,
                    frame_shape=preview_frame.image.shape,
                )
            except Exception as exc:
                print(f"[ERROR] 甯у鐞嗗け璐? {exc}")
                if self._detection_logger is not None:
                    self._detection_logger.log_frame(
                        {
                            "frame_index": int(self._inference_frame_index),
                            "status": "frame_process_error",
                            "error": str(exc),
                        }
                    )
                time.sleep(0.01)

    def _frame_to_qimage(self, frame: np.ndarray) -> QImage:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        return QImage(
            rgb_frame.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
