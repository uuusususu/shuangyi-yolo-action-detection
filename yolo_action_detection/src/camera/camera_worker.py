"""Camera worker module - coordinates camera capture and inference."""
from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

from camera.mvsdk_camera import MvSdkCamera, MvSdkCapture, MvSdkDevice
from yolo_runtime.yolo_result_models import DetectionOverlayState, PipelineStats


class CameraWorker(QObject):
    """Coordinate camera capture, inference scheduling, and runtime events."""

    frame_ready = Signal(QImage)
    error_occurred = Signal(str)
    state_changed = Signal(str, bool)
    fps_updated = Signal(float)
    overlay_state_updated = Signal(dict)
    pipeline_stats_updated = Signal(dict)
    camera_params_updated = Signal(dict)

    def __init__(self, config, state) -> None:
        super().__init__()
        self.config = config
        self.state = state
        self.cap: Optional[object] = None
        self.running = False
        self.frame_processor = None

        self._mvsdk_camera: Optional[MvSdkCamera] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None
        self._session_generation = 0

        self._preview_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._overlay_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._display_lock = threading.Lock()

        self._latest_preview_frame: Optional[np.ndarray] = None
        self._latest_infer_frame: Optional[np.ndarray] = None
        self._latest_infer_frame_id: int = 0
        self._latest_overlay = DetectionOverlayState()
        self._pipeline_stats = PipelineStats()

        # 同帧推理显示：保存已绘制 overlay 的帧
        self._latest_display_frame: Optional[np.ndarray] = None
        self._latest_display_frame_id: int = 0

        self._next_frame_id = 0
        self._last_inferred_frame_id = 0

        self._capture_count = 0
        self._capture_fps_start = time.time()
        self._infer_count = 0
        self._infer_fps_start = time.time()
        self._preview_count = 0
        self._preview_fps_start = time.time()

    def set_frame_processor(self, processor) -> None:
        self.frame_processor = processor

    @staticmethod
    def enumerate_devices() -> list[MvSdkDevice]:
        return MvSdkCamera.enumerate_devices()

    @property
    def active_device(self) -> Optional[MvSdkDevice]:
        if self._mvsdk_camera is None:
            return None
        return self._mvsdk_camera.device

    def open_camera(self) -> bool:
        """Open the explicitly configured MvSDK camera."""
        self._clear_session_buffers()
        try:
            target_sn = str(getattr(self.config, "mvsdk_camera_sn", "") or "").strip()
            legacy_name = str(
                getattr(self.config, "mvsdk_friendly_name", "") or ""
            ).strip()
            if not target_sn and not legacy_name:
                raise RuntimeError("未选择迈德相机，请先在配置页选择并保存")

            self._mvsdk_camera = MvSdkCamera(
                exposure_us=getattr(self.config, "camera_manual_exposure_us", 30000),
                auto_exposure=getattr(self.config, "camera_manual_ae", False),
            )
            param_mode = getattr(self.config, "camera_parameter_mode", "preserve")
            param_group = getattr(self.config, "camera_parameter_group", 0)
            param_file = getattr(self.config, "camera_parameter_file", "")
            opened = self._mvsdk_camera.open(
                sn=target_sn,
                friendly_name=legacy_name if not target_sn else "",
                parameter_mode=param_mode,
                parameter_group=param_group,
                parameter_file=param_file,
            )
            if not opened:
                raise RuntimeError(
                    self._mvsdk_camera.last_error
                    or f"无法打开迈德相机: {target_sn or legacy_name}"
                )
            self.cap = MvSdkCapture(self._mvsdk_camera)
            try:
                params = self._mvsdk_camera.read_params()
                self.camera_params_updated.emit(params)
            except Exception:
                pass

            self.running = True
            self.state.set_camera_on(True)
            self._start_threads()
            return True
        except Exception as exc:
            if self._mvsdk_camera is not None:
                try:
                    self._mvsdk_camera.close()
                except Exception:
                    pass
            self._mvsdk_camera = None
            self.cap = None
            self.running = False
            self.state.set_camera_on(False)
            self.state.set_inference_on(False)
            self.error_occurred.emit(str(exc))
            return False

    def close_camera(self) -> None:
        self.running = False
        self._session_generation += 1
        self.state.set_camera_on(False)
        self.state.set_inference_on(False)
        self._wait_for_threads_to_stop()
        if self._mvsdk_camera:
            try:
                self._mvsdk_camera.close()
            except Exception:
                pass
            self._mvsdk_camera = None
        elif self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self._clear_session_buffers()

    def _wait_for_threads_to_stop(self, timeout_s: float = 1.0) -> None:
        current_thread = threading.current_thread()
        for attr_name in ("_capture_thread", "_inference_thread"):
            thread = getattr(self, attr_name)
            if thread is not None and thread is not current_thread and thread.is_alive():
                thread.join(timeout=timeout_s)
            if thread is None or not thread.is_alive():
                setattr(self, attr_name, None)

    def start_inference(self) -> None:
        self.state.set_inference_on(True)

    def stop_inference(self) -> None:
        self.state.set_inference_on(False)

    def _start_threads(self) -> None:
        self._session_generation += 1
        generation = self._session_generation
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            args=(generation,),
            daemon=True,
        )
        self._capture_thread.start()
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            args=(generation,),
            daemon=True,
        )
        self._inference_thread.start()

    def _capture_loop(self, generation: int) -> None:
        while self.running and generation == self._session_generation and self.cap:
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                if not self.running or generation != self._session_generation:
                    break
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                with self._preview_lock:
                    self._next_frame_id += 1
                    frame_id = self._next_frame_id
                    self._latest_preview_frame = frame.copy()
                with self._infer_lock:
                    self._latest_infer_frame = frame.copy()
                    self._latest_infer_frame_id = frame_id
                self._capture_count += 1
                self._update_capture_fps()
            except Exception:
                time.sleep(0.05)

    def _inference_loop(self, generation: int) -> None:
        while self.running and generation == self._session_generation:
            if not self.state.inference_on or self.frame_processor is None:
                time.sleep(0.05)
                continue
            with self._infer_lock:
                frame = self._latest_infer_frame
                fid = self._latest_infer_frame_id
            if frame is None:
                time.sleep(0.01)
                continue
            if fid <= self._last_inferred_frame_id:
                time.sleep(0.005)
                continue
            dropped = max(0, fid - self._last_inferred_frame_id - 1)
            self._last_inferred_frame_id = fid
            try:
                overlay = self.frame_processor.process_frame(frame, source_frame_id=fid)
                if not self.running or generation != self._session_generation:
                    break
                with self._overlay_lock:
                    self._latest_overlay = overlay
                # 同帧：保存推理完成的帧和 overlay
                with self._display_lock:
                    self._latest_display_frame = frame.copy()
                    self._latest_display_frame_id = fid
                self.overlay_state_updated.emit(self._overlay_to_dict(overlay))
                self._infer_count += 1
                if dropped:
                    self._pipeline_stats.dropped_for_infer += dropped
                self._pipeline_stats.infer_latency_ms = float(overlay.latency_ms or 0.0)
                self._update_infer_fps()
            except Exception as exc:
                self.error_occurred.emit(f"推理异常: {exc}")

    def _clear_session_buffers(self) -> None:
        with self._preview_lock:
            self._latest_preview_frame = None
            self._next_frame_id = 0
        with self._infer_lock:
            self._latest_infer_frame = None
            self._latest_infer_frame_id = 0
            self._last_inferred_frame_id = 0
        with self._display_lock:
            self._latest_display_frame = None
            self._latest_display_frame_id = 0
        with self._overlay_lock:
            self._latest_overlay = DetectionOverlayState()
        with self._stats_lock:
            self._pipeline_stats = PipelineStats()
            self._capture_count = 0
            self._infer_count = 0
            self._preview_count = 0
            now = time.time()
            self._capture_fps_start = now
            self._infer_fps_start = now
            self._preview_fps_start = now

    def get_latest_preview_frame(self) -> Optional[np.ndarray]:
        with self._preview_lock:
            return self._latest_preview_frame.copy() if self._latest_preview_frame is not None else None

    def get_latest_preview_frame_id(self) -> int:
        with self._preview_lock:
            return int(self._next_frame_id)

    def get_display_frame(self) -> tuple:
        """获取同帧推理结果（frame, frame_id）。检测模式下使用。"""
        with self._display_lock:
            if self._latest_display_frame is not None:
                return self._latest_display_frame.copy(), self._latest_display_frame_id
        return None, 0

    def get_camera_params(self) -> dict:
        """读取当前相机参数。"""
        if self._mvsdk_camera:
            return self._mvsdk_camera.read_params()
        return {"read_ok": False, "error": "非 MvSDK 相机"}

    def get_latest_overlay(self) -> DetectionOverlayState:
        with self._overlay_lock:
            return self._latest_overlay

    def get_pipeline_stats(self) -> dict:
        return self._pipeline_stats.to_dict()

    def _update_capture_fps(self) -> None:
        now = time.time()
        elapsed = now - self._capture_fps_start
        if elapsed >= 1.0:
            self._pipeline_stats.capture_fps = self._capture_count / elapsed
            self._capture_count = 0
            self._capture_fps_start = now

    def _update_infer_fps(self) -> None:
        now = time.time()
        elapsed = now - self._infer_fps_start
        if elapsed >= 1.0:
            self._pipeline_stats.infer_fps = self._infer_count / elapsed
            self._infer_count = 0
            self._infer_fps_start = now
            self.pipeline_stats_updated.emit(self._pipeline_stats.to_dict())

    @staticmethod
    def _overlay_to_dict(overlay: DetectionOverlayState) -> dict:
        dets = []
        for d in overlay.detections:
            det = d.to_dict()
            det["center"] = det["center_px"]
            dets.append(det)
        return {
            "source_frame_id": overlay.source_frame_id,
            "timestamp": overlay.timestamp,
            "model_path": overlay.model_path,
            "task_type": overlay.task_type,
            "detections": dets,
            "status": overlay.status,
            "error": overlay.error,
            "round_id": overlay.round_id,
            "latency_ms": overlay.latency_ms,
        }
