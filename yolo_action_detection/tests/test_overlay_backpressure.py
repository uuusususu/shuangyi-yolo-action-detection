"""推理结果送界面的积压上限：不无限堆帧，也不在持锁时等界面，停止时立即唤醒。"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app_state import AppState
from camera.camera_worker import CameraWorker
from config import ConfigManager
from yolo_runtime.yolo_result_models import DetectionOverlayState


class _Processor:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    def process_frame(self, frame, source_frame_id=0):
        self.calls += 1
        return DetectionOverlayState(source_frame_id=source_frame_id)

    def close(self):
        self.closed = True


def _worker_with_frames():
    worker = CameraWorker(ConfigManager(), AppState())
    processor = _Processor()
    worker.frame_processor = processor
    worker.running = True
    worker.state.set_camera_on(True)
    worker.state.set_inference_on(True)
    emitted = []
    # 用例里没有界面事件循环，直连让推理线程发出时立即记录。
    worker.overlay_state_updated.connect(
        lambda payload: emitted.append(payload["source_frame_id"]),
        Qt.ConnectionType.DirectConnection,
    )
    return worker, processor, emitted


def _feed(worker, frame_id):
    with worker._infer_lock:
        worker._latest_infer_frame = np.zeros((4, 4, 3), dtype=np.uint8)
        worker._latest_infer_frame_id = frame_id
    worker._frame_ready.set()


def _start_inference(worker):
    thread = threading.Thread(
        target=worker._inference_loop, args=(worker._session_generation,), daemon=True
    )
    thread.start()
    return thread


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    return predicate()


def test_unacknowledged_results_are_capped():
    worker, processor, emitted = _worker_with_frames()
    worker._OVERLAY_WAIT_LIMIT_S = 5.0
    thread = _start_inference(worker)
    try:
        for frame_id in range(1, 20):
            _feed(worker, frame_id)
            time.sleep(0.01)
        assert _wait_until(lambda: len(emitted) >= worker._OVERLAY_BACKLOG_LIMIT)
        time.sleep(0.1)
        # 界面一直没确认：最多发出上限条，不会把后面的帧都堆进队列。
        assert len(emitted) == worker._OVERLAY_BACKLOG_LIMIT

        worker.ack_overlay_event()
        assert _wait_until(lambda: len(emitted) == worker._OVERLAY_BACKLOG_LIMIT + 1)
        # 确认后推的是最新帧，中间等待期间的旧帧被跳过。
        assert emitted[-1] == 19
    finally:
        worker.running = False
        worker._wake_overlay_waiters()
        thread.join(timeout=2.0)


def test_gui_can_swap_processor_while_inference_waits_for_ack():
    worker, _processor, emitted = _worker_with_frames()
    worker._OVERLAY_WAIT_LIMIT_S = 5.0
    thread = _start_inference(worker)
    try:
        for frame_id in range(1, 10):
            _feed(worker, frame_id)
            time.sleep(0.01)
        assert _wait_until(lambda: len(emitted) >= worker._OVERLAY_BACKLOG_LIMIT)
        time.sleep(0.05)
        # 推理线程此刻在等确认；界面线程换处理器不能被它持锁卡住。
        replacement = _Processor()
        started = time.monotonic()
        worker.set_frame_processor(replacement)
        assert time.monotonic() - started < 0.5
        assert _processor.closed is True
    finally:
        worker.running = False
        worker._wake_overlay_waiters()
        thread.join(timeout=2.0)


def test_stop_wakes_a_waiting_inference_thread_immediately():
    worker, _processor, emitted = _worker_with_frames()
    worker._OVERLAY_WAIT_LIMIT_S = 5.0
    thread = _start_inference(worker)
    for frame_id in range(1, 10):
        _feed(worker, frame_id)
        time.sleep(0.01)
    assert _wait_until(lambda: len(emitted) >= worker._OVERLAY_BACKLOG_LIMIT)
    time.sleep(0.05)

    started = time.monotonic()
    worker.running = False
    worker._wake_overlay_waiters()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    # 远小于 5 秒的等待上限：停止不需要等超时。
    assert time.monotonic() - started < 0.5
