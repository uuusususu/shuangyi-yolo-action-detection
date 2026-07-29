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
from runtime_diagnostics import get_runtime_logger
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
        self._runtime_logger = get_runtime_logger()
        self._first_frame_logged = False
        self._first_inference_logged = False
        self._capture_wait_started_at: Optional[float] = None
        self._capture_wait_warning_logged = False
        self._logged_inference_errors: set[str] = set()
        self._last_capture_exception = ""

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
        self._runtime_logger.info(
            "frame_processor=set type=%s",
            type(processor).__name__ if processor is not None else "None",
        )

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
            self._runtime_logger.info(
                "camera_open=attempt target_sn=%s legacy_name=%s",
                target_sn or "unset",
                legacy_name or "unset",
            )

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
            active_device = self.active_device
            self._runtime_logger.info(
                "camera_open=ok active_sn=%s active_name=%s",
                getattr(active_device, "sn", "unknown"),
                getattr(active_device, "friendly_name", "unknown"),
            )
            return True
        except Exception as exc:
            self._runtime_logger.exception("camera_open=failed error=%s", exc)
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
        self._runtime_logger.info(
            "camera_close capture_frames=%s inference_frames=%s",
            self._capture_count,
            self._infer_count,
        )
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
        self._runtime_logger.info(
            "inference_state=start camera_on=%s processor=%s",
            self.state.camera_on,
            type(self.frame_processor).__name__ if self.frame_processor is not None else "None",
        )
        self.state.set_inference_on(True)

    def stop_inference(self) -> None:
        self._runtime_logger.info("inference_state=stop")
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
                    now = time.monotonic()
                    if self._capture_wait_started_at is None:
                        self._capture_wait_started_at = now
                    elif (
                        not self._capture_wait_warning_logged
                        and now - self._capture_wait_started_at >= 3.0
                    ):
                        self._runtime_logger.error(
                            "camera_frames=unavailable duration_seconds=3"
                        )
                        self._capture_wait_warning_logged = True
                    time.sleep(0.01)
                    continue
                if not self.running or generation != self._session_generation:
                    break
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                self._capture_wait_started_at = None
                self._capture_wait_warning_logged = False
                if not self._first_frame_logged:
                    self._runtime_logger.info(
                        "camera_first_frame=ok shape=%s dtype=%s",
                        tuple(frame.shape),
                        frame.dtype,
                    )
                    self._first_frame_logged = True
                with self._preview_lock:
                    self._next_frame_id += 1
                    frame_id = self._next_frame_id
                    self._latest_preview_frame = frame.copy()
                with self._infer_lock:
                    self._latest_infer_frame = frame.copy()
                    self._latest_infer_frame_id = frame_id
                self._capture_count += 1
                self._update_capture_fps()
            except Exception as exc:
                error_key = f"{type(exc).__name__}:{exc}"
                if error_key != self._last_capture_exception:
                    self._runtime_logger.exception(
                        "camera_capture=failed error=%s",
                        exc,
                    )
                    self._last_capture_exception = error_key
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
                self._record_inference_diagnostics(overlay)
                if not self.running or generation != self._session_generation:
                    break
                with self._overlay_lock:
                    self._latest_overlay = overlay
                # 同帧：保存推理完成的帧和 overlay
                with self._display_lock:
                    self._latest_display_frame = frame.copy()
                    self._latest_display_frame_id = fid
                image_size = (int(frame.shape[1]), int(frame.shape[0]))
                self.overlay_state_updated.emit(
                    self._overlay_to_dict(overlay, image_size=image_size)
                )
                self._infer_count += 1
                if dropped:
                    self._pipeline_stats.dropped_for_infer += dropped
                self._pipeline_stats.infer_latency_ms = float(overlay.latency_ms or 0.0)
                self._update_infer_fps()
            except Exception as exc:
                error_key = f"loop:{type(exc).__name__}:{exc}"
                if error_key not in self._logged_inference_errors:
                    self._runtime_logger.exception(
                        "inference_loop=failed frame_id=%s error=%s",
                        fid,
                        exc,
                    )
                    self._logged_inference_errors.add(error_key)
                self.error_occurred.emit(f"推理异常: {exc}")

    def _record_inference_diagnostics(self, overlay: DetectionOverlayState) -> None:
        if overlay.error:
            error_key = str(overlay.error)
            if error_key not in self._logged_inference_errors:
                self._runtime_logger.error(
                    "inference_result=failed frame_id=%s model=%s error=%s",
                    overlay.source_frame_id,
                    overlay.model_path,
                    overlay.error,
                )
                self._logged_inference_errors.add(error_key)
            return
        if not self._first_inference_logged:
            self._runtime_logger.info(
                "inference_first_result=ok frame_id=%s detections=%s latency_ms=%s",
                overlay.source_frame_id,
                len(overlay.detections),
                overlay.latency_ms,
            )
            self._first_inference_logged = True

    def _clear_session_buffers(self) -> None:
        self._first_frame_logged = False
        self._first_inference_logged = False
        self._capture_wait_started_at = None
        self._capture_wait_warning_logged = False
        self._logged_inference_errors.clear()
        self._last_capture_exception = ""
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
    def _overlay_to_dict(
        overlay: DetectionOverlayState,
        *,
        image_size: tuple[int, int],
    ) -> dict:
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
            "image_size": (int(image_size[0]), int(image_size[1])),
            "status": overlay.status,
            "error": overlay.error,
            "round_id": overlay.round_id,
            "latency_ms": overlay.latency_ms,
            "template_matched": bool(overlay.template_matched),
            "template_score": float(overlay.template_score),
            "template_center": list(overlay.template_center) if overlay.template_center else None,
            "template_angle": float(overlay.template_angle) if overlay.template_angle is not None else None,
        }
