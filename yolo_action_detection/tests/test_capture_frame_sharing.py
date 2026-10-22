from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app_state import AppState
from camera.camera_worker import CameraWorker
from config import ConfigManager


class _FakeCapture:
    """按次数产出独立帧，用完后表现为暂时无帧。"""

    def __init__(self, frame_limit: int) -> None:
        self._frame_limit = frame_limit
        self.produced = 0

    def read(self):
        if self.produced >= self._frame_limit:
            time.sleep(0.001)
            return False, None
        self.produced += 1
        return True, np.full((8, 10, 3), self.produced, dtype=np.uint8)


def _run_capture(worker: CameraWorker, capture: _FakeCapture) -> None:
    worker.cap = capture
    worker.running = True
    generation = worker._session_generation
    thread = threading.Thread(
        target=worker._capture_loop, args=(generation,), daemon=True
    )
    thread.start()
    deadline = time.monotonic() + 2.0
    while capture.produced < capture._frame_limit and time.monotonic() < deadline:
        time.sleep(0.005)
    time.sleep(0.02)
    worker.running = False
    thread.join(timeout=1.0)


def _worker() -> CameraWorker:
    return CameraWorker(ConfigManager(), AppState())


def test_capture_shares_one_readonly_snapshot_between_preview_and_inference():
    worker = _worker()
    capture = _FakeCapture(frame_limit=3)

    _run_capture(worker, capture)

    preview = worker._latest_preview_frame
    infer = worker._latest_infer_frame
    assert preview is not None
    assert preview is infer
    assert preview.flags.writeable is False
    assert worker._latest_infer_frame_id == worker._next_frame_id


def test_consumers_receive_writable_copies_of_shared_snapshot():
    worker = _worker()
    capture = _FakeCapture(frame_limit=2)

    _run_capture(worker, capture)

    shared = worker._latest_preview_frame
    copy = worker.get_latest_preview_frame()

    assert copy is not shared
    assert copy.flags.writeable is True
    copy[0, 0] = 200
    assert shared[0, 0, 0] != 200


def test_drawing_on_display_frame_leaves_shared_snapshot_untouched():
    """界面拿到的是副本，在上面画框不能反过来改掉推理或预览持有的那一份。"""
    import cv2

    worker = _worker()
    capture = _FakeCapture(frame_limit=2)
    _run_capture(worker, capture)

    shared = worker._latest_infer_frame
    with worker._display_lock:
        worker._latest_display_frame = shared
        worker._latest_display_frame_id = worker._latest_infer_frame_id
    before = shared.copy()

    frame, _frame_id = worker.get_display_frame()
    cv2.rectangle(frame, (0, 0), (9, 7), (0, 255, 0), 2)

    assert np.array_equal(shared, before)
    assert np.array_equal(worker._latest_preview_frame, before)


def test_display_snapshot_is_shared_not_recopied():
    worker = _worker()
    frame = np.zeros((8, 10, 3), dtype=np.uint8)
    frame.flags.writeable = False

    with worker._display_lock:
        worker._latest_display_frame = frame
        worker._latest_display_frame_id = 5

    assert worker.get_display_frame_id() == 5
    returned, frame_id = worker.get_display_frame()
    assert frame_id == 5
    assert returned is not frame
    assert returned.flags.writeable is True
