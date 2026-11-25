"""Camera worker module - coordinates camera capture and inference."""
from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from copy import deepcopy
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from camera.mvsdk_camera import MvSdkCamera, MvSdkCapture, MvSdkDevice
from hand_tracking.mediapipe_hands import landmarks_to_payload
from runtime_diagnostics import get_runtime_logger
from yolo_runtime.yolo_result_models import DetectionOverlayState, PipelineStats


class CameraWorker(QObject):
    """Coordinate camera capture, inference scheduling, and runtime events."""

    error_occurred = Signal(str)
    state_changed = Signal(str, bool)
    fps_updated = Signal(float)
    overlay_state_updated = Signal(dict)
    pipeline_stats_updated = Signal(dict)
    camera_params_updated = Signal(dict)
    _ACTION_EVIDENCE_LIMIT = 16
    _OVERLAY_BACKLOG_LIMIT = 4
    _OVERLAY_WAIT_LIMIT_S = 1.0
    _DEFAULT_MAX_INFER_FPS = 10.0
    _DEFAULT_MIN_COOLDOWN_S = 0.025

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
        self._runtime_lock = threading.RLock()
        self.runtime_generation = 0
        self._evidence_lock = threading.Lock()
        self._action_evidence = OrderedDict()
        self._action_event_evidence = deque(maxlen=4)

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
        # 采集到新帧时唤醒推理线程，替代固定间隔轮询。
        self._frame_ready = threading.Event()
        # 推理帧率上限与最小帧间冷却：防止低核 CPU（如 i3-7100）被背靠背推理打满 100%。
        self._max_infer_fps = self._DEFAULT_MAX_INFER_FPS
        self._min_infer_cooldown_s = self._DEFAULT_MIN_COOLDOWN_S
        self._stop_pacing = threading.Event()
        # 已发出、界面还没处理完的推理结果数量。
        self._overlay_backlog = 0
        self._overlay_backlog_cond = threading.Condition()

        self._capture_count = 0
        self._capture_fps_start = time.time()
        self._infer_count = 0
        self._infer_fps_start = time.time()
        self._preview_count = 0
        self._preview_fps_start = time.time()

        self._hand_lock = threading.Lock()
        self._hand_overlay_enabled = False
        self._hand_tracker = None
        self._hand_tracker_failed = False
        self._hand_runtime_error = ""
        self._hand_tracker_factory = self._create_runtime_hands
        self._action_lock = threading.Lock()
        self._action_session = None

    @staticmethod
    def _create_runtime_hands():
        from lstm_workbench.model_registry import create_hands_tracker

        return create_hands_tracker()

    def get_action_session(self):
        with self._action_lock:
            return self._action_session

    def replace_runtime(self, config, processor, action_session) -> None:
        """在停推理状态一次切换已准备好的依赖，不在提交阶段装载模型。"""
        with self._runtime_lock:
            if self.state.inference_on:
                raise RuntimeError("请先停止推理，再切换模型配置")
            with self._action_lock:
                self.config = config
                previous = self.frame_processor
                self.frame_processor = processor
                self._action_session = action_session
                self.runtime_generation += 1
            self._close_processor(previous, keep=processor)
            with self._hand_lock:
                self._release_hand_tracker_locked()
                self._hand_tracker_failed = False
                self._hand_runtime_error = ""
            self._clear_action_evidence()

    def set_frame_processor(self, processor) -> None:
        # 推理线程持 _runtime_lock 跑完整帧，这里等它结束再换，不会半帧换处理器。
        with self._runtime_lock:
            previous = self.frame_processor
            self.frame_processor = processor
        self._close_processor(previous, keep=processor)
        self._runtime_logger.info(
            "frame_processor=set type=%s",
            type(processor).__name__ if processor is not None else "None",
        )

    def _close_processor(self, processor, *, keep) -> None:
        """显式释放被替换下来的处理器；同一个对象继续使用时不动。"""
        if processor is None or processor is keep:
            return
        close = getattr(processor, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception:
            self._runtime_logger.warning("frame_processor=close_failed", exc_info=True)

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
            opened = self._mvsdk_camera.open(
                sn=target_sn,
                friendly_name=legacy_name if not target_sn else "",
                parameter_mode="preserve",
                parameter_group=0,
                parameter_file="",
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
                self._runtime_logger.warning("camera_params=read_failed", exc_info=True)

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
        self._frame_ready.set()
        # 推理线程可能正在等界面确认积压；running 已为 False，叫醒它马上退出。
        self._wake_overlay_waiters()
        if not self._wait_for_threads_to_stop():
            # 采集线程还没退出：MvSdkCamera.close 会等它当前这次 read 结束再释放句柄。
            self._runtime_logger.warning("camera_close=threads_still_running")
        if self._mvsdk_camera:
            try:
                self._mvsdk_camera.close()
            except Exception:
                self._runtime_logger.warning("camera_close=sdk_close_failed", exc_info=True)
            self._mvsdk_camera = None
        elif self.cap:
            try:
                self.cap.release()
            except Exception:
                self._runtime_logger.warning("camera_close=release_failed", exc_info=True)
        self.cap = None
        self._release_hand_tracker()
        self._clear_session_buffers()

    def _wait_for_threads_to_stop(self, timeout_s: float = 1.0) -> bool:
        """返回两个线程是否都已退出。"""
        current_thread = threading.current_thread()
        all_stopped = True
        for attr_name in ("_capture_thread", "_inference_thread"):
            thread = getattr(self, attr_name)
            if thread is not None and thread is not current_thread and thread.is_alive():
                thread.join(timeout=timeout_s)
            if thread is None or not thread.is_alive():
                setattr(self, attr_name, None)
            elif thread is not current_thread:
                all_stopped = False
        return all_stopped

    def set_hand_overlay_enabled(self, enabled: bool) -> None:
        """顺序模式检测中才开 Hands。关闭时释放会话，便于下次重新初始化。"""
        enabled = bool(enabled)
        with self._hand_lock:
            self._hand_overlay_enabled = enabled
            if not enabled:
                self._release_hand_tracker_locked()
                self._hand_tracker_failed = False
                self._hand_runtime_error = ""

    def start_inference(self) -> None:
        self._runtime_logger.info(
            "inference_state=start camera_on=%s processor=%s",
            self.state.camera_on,
            type(self.frame_processor).__name__ if self.frame_processor is not None else "None",
        )
        self._stop_pacing.clear()
        self.state.set_inference_on(True)

    def stop_inference(self) -> None:
        self._runtime_logger.info("inference_state=stop")
        self.state.set_inference_on(False)
        self._stop_pacing.set()
        self._wake_overlay_waiters()

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
                read_started = time.perf_counter()
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
                # 相机层已经返回独立副本，这里把同一份快照标记为只读后共享给预览和推理，
                # 省掉每帧两次整幅拷贝；取用方各自 copy，写入方只换引用不改内容。
                frame.flags.writeable = False
                with self._preview_lock:
                    self._next_frame_id += 1
                    frame_id = self._next_frame_id
                    self._latest_preview_frame = frame
                with self._infer_lock:
                    self._latest_infer_frame = frame
                    self._latest_infer_frame_id = frame_id
                    self._latest_infer_captured_at = time.monotonic()
                    self._latest_infer_captured_wall_time = time.time()
                self._frame_ready.set()
                self._pipeline_stats.capture_ms = (
                    time.perf_counter() - read_started
                ) * 1000.0
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
                captured_at = getattr(self, "_latest_infer_captured_at", None)
                captured_wall_time = getattr(self, "_latest_infer_captured_wall_time", None)
            if frame is None or fid <= self._last_inferred_frame_id:
                # 先清再复查，避免清掉采集线程刚发出的通知；超时兜底防止漏唤醒。
                self._frame_ready.clear()
                with self._infer_lock:
                    has_new_frame = (
                        self._latest_infer_frame is not None
                        and self._latest_infer_frame_id > self._last_inferred_frame_id
                    )
                if not has_new_frame:
                    self._frame_ready.wait(0.05)
                continue
            # 积压等待放在拿 _runtime_lock 之前：界面线程换处理器、停推理时也要拿这把锁，
            # 若在锁内等界面确认，界面拿不到锁又无法确认，只能干等超时。
            self._wait_for_overlay_capacity()
            if not self.running or generation != self._session_generation:
                break
            if not self.state.inference_on:
                continue
            with self._infer_lock:
                # 等待期间可能来了更新的帧，直接推最新的。
                if self._latest_infer_frame is not None and self._latest_infer_frame_id > fid:
                    frame = self._latest_infer_frame
                    fid = self._latest_infer_frame_id
                    captured_at = getattr(self, "_latest_infer_captured_at", None)
                    captured_wall_time = getattr(self, "_latest_infer_captured_wall_time", None)
            dropped = max(0, fid - self._last_inferred_frame_id - 1)
            self._last_inferred_frame_id = fid
            loop_started = time.monotonic()
            self._runtime_lock.acquire()
            try:
                if not self.state.inference_on:
                    continue
                overlay = self.frame_processor.process_frame(frame, source_frame_id=fid)
                overlay.source_frame_id = int(fid)
                self._record_inference_diagnostics(overlay)
                if not self.running or generation != self._session_generation:
                    break
                action_session = self.get_action_session()
                uses_hands = bool(getattr(getattr(action_session, "bundle", None), "hands_in_training", False))
                hand_landmarks = self._hand_landmarks_payload(
                    frame,
                    force=uses_hands,
                )
                if hand_landmarks is not None:
                    overlay.hand_landmarks = hand_landmarks
                detections = self._action_detections(frame, overlay, fid)
                observed_at = float(captured_at if captured_at is not None else time.monotonic())
                self._cache_action_evidence(
                    frame, fid, observed_at, detections, overlay.model_path,
                    wall_time=captured_wall_time,
                )
                action_report = self._observe_action(
                    frame,
                    fid,
                    observed_at,
                    overlay,
                    hand_landmarks,
                    detections=detections,
                )
                self._prune_action_evidence(action_report)
                self._publish_inference_result(frame, fid, overlay, action_report)
                self._infer_count += 1
                if dropped:
                    self._pipeline_stats.dropped_for_infer += dropped
                self._pipeline_stats.infer_latency_ms = float(overlay.latency_ms or 0.0)
                self._pipeline_stats.template_ms = float(overlay.template_ms or 0.0)
                self._pipeline_stats.yolo_ms = float(overlay.yolo_ms or 0.0)
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
                action_report = self._fault_action(str(exc))
                self._prune_action_evidence(action_report)
                if self.running and generation == self._session_generation:
                    failed = DetectionOverlayState(source_frame_id=fid, error=str(exc), status="error")
                    self._publish_inference_result(frame, fid, failed, action_report)
                self.error_occurred.emit(f"推理异常: {exc}")
            finally:
                self._runtime_lock.release()
            self._pace_inference_loop(loop_started)

    def _pace_inference_loop(self, loop_started: float) -> None:
        """每帧推理后按帧率上限和最小冷却主动让步，防止低核工控机 CPU 持续 100%。"""
        if not self.running or not self.state.inference_on:
            return
        elapsed = max(0.0, time.monotonic() - loop_started)
        min_period = (1.0 / self._max_infer_fps) if self._max_infer_fps > 0 else 0.0
        sleep_for = max(self._min_infer_cooldown_s, min_period - elapsed)
        if sleep_for > 0.0:
            self._stop_pacing.wait(sleep_for)

    def _publish_inference_result(self, frame, frame_id: int, overlay, action_report) -> None:
        """同一次输入的图片、帧号和结果一起发布，包括故障结果。"""
        overlay.source_frame_id = int(frame_id)
        with self._overlay_lock:
            self._latest_overlay = overlay
        with self._display_lock:
            self._latest_display_frame = frame
            self._latest_display_frame_id = int(frame_id)
        payload = self._overlay_to_dict(overlay, image_size=(frame.shape[1], frame.shape[0]))
        if overlay.hand_landmarks:
            payload["hand_landmarks"] = overlay.hand_landmarks
        if action_report is not None:
            report = dict(action_report)
            events = []
            for original in action_report.get("events", []):
                event = dict(original)
                evidence = self.get_action_boundary_evidence(int(event.get("end_frame") or 0))
                if evidence is not None:
                    image, detections, metadata = evidence
                    metadata["runtime_generation"] = self.runtime_generation
                    # 图片归该 Qt 消息所有，缓存驱逐不能使尚未消费的 NG 丢图。
                    event["_boundary_evidence"] = (image, detections, metadata)
                events.append(event)
            report["events"] = events
            payload["lstm_action"] = report
        payload["source_frame"] = frame
        payload["runtime_generation"] = self.runtime_generation
        with self._overlay_backlog_cond:
            self._overlay_backlog += 1
        self.overlay_state_updated.emit(payload)

    def _wait_for_overlay_capacity(self) -> None:
        """界面积压时让推理线程等一等，而不是把整帧图片无限堆进 Qt 事件队列。

        判定仍逐帧推进；推理慢下来后多出的相机帧在"只取最新帧"这一步被丢弃。
        等待有上限，界面卡死或没人确认时照常发出，避免推理线程永久阻塞。
        """
        deadline = time.monotonic() + self._OVERLAY_WAIT_LIMIT_S
        with self._overlay_backlog_cond:
            while (
                self._overlay_backlog >= self._OVERLAY_BACKLOG_LIMIT
                and self.running
                and self.state.inference_on
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._pipeline_stats.overlay_backlog_timeouts += 1
                    break
                self._overlay_backlog_cond.wait(remaining)

    def _wake_overlay_waiters(self) -> None:
        self._stop_pacing.set()
        with self._overlay_backlog_cond:
            self._overlay_backlog_cond.notify_all()

    def ack_overlay_event(self) -> None:
        """界面处理完一条 overlay_state_updated 后调用，释放一个积压名额。"""
        with self._overlay_backlog_cond:
            self._overlay_backlog = max(0, self._overlay_backlog - 1)
            self._overlay_backlog_cond.notify()

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
        self.runtime_generation += 1
        self._clear_action_evidence()
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
        # 新会话不继承上一会话没确认的积压，否则会被旧事件卡住。
        with self._overlay_backlog_cond:
            self._overlay_backlog = 0
            self._overlay_backlog_cond.notify_all()

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

    def get_display_frame_size(self) -> tuple:
        """只取同帧推理结果的 (宽, 高)，不拷贝整幅图像。没有帧时返回空元组。"""
        with self._display_lock:
            if self._latest_display_frame is None:
                return ()
            height, width = self._latest_display_frame.shape[:2]
        return (int(width), int(height))

    def get_display_frame_id(self) -> int:
        """只取同帧推理结果的帧号，供界面在拷贝前判断内容是否变化。"""
        with self._display_lock:
            if self._latest_display_frame is None:
                return 0
            return int(self._latest_display_frame_id)

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

    def record_render_latency(self, elapsed_ms: float) -> None:
        """界面线程回报一次渲染耗时，让分段统计覆盖到显示这一段。"""
        with self._stats_lock:
            self._pipeline_stats.render_ms = float(elapsed_ms)

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

    def set_action_session(self, session) -> None:
        """推理线程持有连续动作会话。传 None 就停止动作推理。"""
        with self._runtime_lock:
            with self._action_lock:
                self._action_session = session
                self.runtime_generation += 1
            self._clear_action_evidence()

    def reset_action_session(self) -> None:
        """停止或关相机时清掉未结束的动作，不把尾段补成 PASS。"""
        with self._runtime_lock:
            with self._action_lock:
                session = self._action_session
            if session is not None:
                session.reset()
            self.runtime_generation += 1
            self._clear_action_evidence()

    def _clear_action_evidence(self) -> None:
        with self._evidence_lock:
            self._action_evidence.clear()
            self._action_event_evidence.clear()

    def _cache_action_evidence(self, frame, frame_id: int, timestamp: float, detections, model_path: str, *, wall_time=None) -> None:
        if self.get_action_session() is None:
            return
        owned = frame.copy()
        owned.setflags(write=False)
        captured_wall = float(wall_time) if wall_time is not None else time.time() + float(timestamp) - time.monotonic()
        with self._evidence_lock:
            self._action_evidence[int(frame_id)] = (
                owned,
                deepcopy(detections),
                {
                    "source_frame_id": int(frame_id), "timestamp": captured_wall,
                    "capture_monotonic": float(timestamp), "model_path": str(model_path),
                },
            )
            while len(self._action_evidence) > self._ACTION_EVIDENCE_LIMIT:
                self._action_evidence.popitem(last=False)

    def _prune_action_evidence(self, report) -> None:
        """仅保留流会话要求的真实边界与最近两帧，长动作不增长图片缓存。"""
        report = report or {}
        keep = {int(value) for value in report.get("evidence_frames", [])}
        with self._evidence_lock:
            for event in report.get("events", []):
                boundaries = {
                    int(event[key]) for key in ("end_frame", "decision_frame")
                    if event.get(key) is not None
                }
                if boundaries:
                    self._action_event_evidence.append(boundaries)
            for boundaries in self._action_event_evidence:
                keep.update(boundaries)
            keep.update(list(self._action_evidence)[-2:])
            for frame_id in list(self._action_evidence):
                if frame_id not in keep:
                    del self._action_evidence[frame_id]

    def get_action_boundary_evidence(self, frame_id: int):
        """半开结束帧指后继的真实源帧；没有该帧就返回 None，不替换图片编号。"""
        with self._evidence_lock:
            evidence = self._action_evidence.get(int(frame_id))
            if evidence is None:
                return None
            frame, detections, metadata = evidence
            return frame.copy(), deepcopy(detections), dict(metadata)

    def _fault_action(self, reason: str):
        session = self.get_action_session()
        if session is None:
            return None
        return session.set_fault(str(reason))

    def _action_detections(self, frame, overlay, frame_id: int):
        """动作特征用整帧 YOLO。模板没对上时不能拿空检测代替。"""
        with self._action_lock:
            if self._action_session is None:
                return getattr(overlay, "detections", []) or []
        if getattr(overlay, "error", ""):
            raise RuntimeError(str(overlay.error))
        inner = getattr(self.frame_processor, "_inner", None)
        if inner is not None and not getattr(overlay, "template_matched", True):
            try:
                detected = inner.process_frame(frame, source_frame_id=frame_id)
            except Exception as exc:
                self._runtime_logger.exception("lstm_action=yolo_failed error=%s", exc)
                raise RuntimeError(f"动作 YOLO 推理失败: {exc}") from exc
            if getattr(detected, "error", ""):
                raise RuntimeError(f"动作 YOLO 推理失败: {detected.error}")
            return getattr(detected, "detections", []) or []
        return getattr(overlay, "detections", []) or []

    def _observe_action(self, frame, frame_id: int, captured_at: float, overlay, hand_landmarks, detections=None):
        with self._action_lock:
            session = self._action_session
        if session is None:
            return None
        if getattr(overlay, "error", ""):
            return self._fault_action(str(overlay.error))
        uses_hands = bool(getattr(getattr(session, "bundle", None), "hands_in_training", True))
        if not uses_hands:
            hand_landmarks = []
        elif hand_landmarks is None:
            hand_landmarks = []
        fault = ""
        if uses_hands and (self._hand_tracker_failed or self._hand_runtime_error):
            fault = self._hand_runtime_error or "手部模型不可用"
        if detections is None:
            detections = []
            for item in getattr(overlay, "detections", []) or []:
                detections.append({
                    "label": getattr(item, "label", ""),
                    "conf": getattr(item, "conf", 0.0),
                    "box": getattr(item, "box", (0, 0, 0, 0)),
                })
        else:
            detections = [
                item if isinstance(item, dict) else {
                    "label": getattr(item, "label", ""),
                    "conf": getattr(item, "conf", 0.0),
                    "box": getattr(item, "box", (0, 0, 0, 0)),
                }
                for item in detections
            ]
        try:
            return session.observe_frame(
                source_frame=int(frame_id),
                timestamp=float(captured_at),
                detections=detections,
                hands=hand_landmarks,
                frame_w=int(frame.shape[1]),
                frame_h=int(frame.shape[0]),
                hand_fault=fault,
            )
        except Exception as exc:
            self._runtime_logger.exception("lstm_action=failed error=%s", exc)
            return self._fault_action(str(exc))

    def _hand_landmarks_payload(self, frame, *, force: bool = False) -> Optional[list]:
        """启用时返回 list（无手为空列表）；未启用返回 None，不往 overlay 写字段。"""
        with self._hand_lock:
            if not self._hand_overlay_enabled and not force:
                return None
            if self._hand_tracker_failed:
                return []
            if self._hand_tracker is None:
                try:
                    tracker = self._hand_tracker_factory()
                except Exception as exc:
                    self._hand_tracker_failed = True
                    self._hand_runtime_error = str(exc)
                    self._runtime_logger.exception(
                        "hand_overlay=factory_failed error=%s",
                        exc,
                    )
                    return []
                if not getattr(tracker, "available", False):
                    self._hand_tracker_failed = True
                    self._hand_runtime_error = getattr(tracker, "error", "") or "手部模型初始化失败"
                    self._runtime_logger.error(
                        "hand_overlay=unavailable error=%s",
                        getattr(tracker, "error", "") or "init_failed",
                    )
                    try:
                        tracker.close()
                    except Exception:
                        pass
                    return []
                self._hand_tracker = tracker
            tracker = self._hand_tracker
            try:
                hands = tracker.process(frame)
                if getattr(tracker, "error", "") or not getattr(tracker, "available", True):
                    raise RuntimeError(getattr(tracker, "error", "") or "手部模型不可用")
            except Exception as exc:
                self._hand_tracker_failed = True
                self._hand_runtime_error = str(exc)
                self._runtime_logger.exception(
                    "hand_overlay=process_failed error=%s",
                    exc,
                )
                return []
        try:
            self._hand_runtime_error = ""
            return landmarks_to_payload(hands)
        except Exception as exc:
            self._hand_tracker_failed = True
            self._hand_runtime_error = f"手部特征转换失败: {exc}"
            return []

    def _release_hand_tracker(self) -> None:
        with self._hand_lock:
            self._release_hand_tracker_locked()

    def _release_hand_tracker_locked(self) -> None:
        tracker = self._hand_tracker
        self._hand_tracker = None
        if tracker is None:
            return
        try:
            tracker.close()
        except Exception:
            pass

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
            "template_ms": float(overlay.template_ms),
            "yolo_ms": float(overlay.yolo_ms),
            "template_matched": bool(overlay.template_matched),
            "template_score": float(overlay.template_score),
            "template_center": list(overlay.template_center) if overlay.template_center else None,
            "template_angle": float(overlay.template_angle) if overlay.template_angle is not None else None,
        }
