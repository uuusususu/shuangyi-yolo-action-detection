from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app_state import AppState
from camera.camera_worker import CameraWorker
from config import ConfigManager
from hand_tracking import mediapipe_hands as hands_mod
from hand_tracking.mediapipe_hands import (
    HandLandmarks,
    MediaPipeHandsTracker,
    draw_hand_landmarks,
    landmarks_to_payload,
)
from step_sequence.step_sequence_engine import StepSequenceEngine
from ui.main_window import MainWindow
from yolo_runtime.yolo_result_models import DetectionOverlayState


def _points(offset: float = 0.0) -> tuple[tuple[float, float], ...]:
    return tuple((10.0 + i * 4.0 + offset, 20.0 + (i % 5) * 6.0) for i in range(21))


class FakeTracker:
    def __init__(self, *, available: bool = True, error: str = "", points=None) -> None:
        self.available = available
        self.error = error
        self.process_calls: list = []
        self.closed = False
        self._hands = [
            HandLandmarks(points=points or _points(), handedness="Right"),
        ]

    def process(self, frame) -> list[HandLandmarks]:
        self.process_calls.append(frame)
        return list(self._hands)

    def close(self) -> None:
        self.closed = True


class DummyProcessor:
    def get_class_names(self):
        return {0: "A", 1: "B"}


class EngineSpy:
    def __init__(self) -> None:
        self.updates: list = []
        self._inner = StepSequenceEngine(step_class_names=["A"], step_counts=[1])
        self._inner.start_round()

    def start_round(self) -> None:
        self._inner.start_round()

    def get_state(self):
        return self._inner.get_state()

    def update(self, detections, frame_id=0):
        self.updates.append((list(detections), frame_id))
        return self._inner.update(detections)


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _window(config: ConfigManager) -> MainWindow:
    return MainWindow(
        config,
        AppState(),
        processor=DummyProcessor(),
        step_engine=StepSequenceEngine(
            step_class_names=["A"],
            step_counts=[1],
        ),
        processor_factory=lambda *args, **kwargs: DummyProcessor(),
    )


def test_tracker_maps_normalized_landmarks_to_pixels(monkeypatch):
    landmark = SimpleNamespace(x=0.25, y=0.5)
    hand = SimpleNamespace(landmark=[landmark] * 21)
    handedness = SimpleNamespace(classification=[SimpleNamespace(label="Left")])
    results = SimpleNamespace(multi_hand_landmarks=[hand], multi_handedness=[handedness])
    fake_hands = SimpleNamespace(process=lambda _rgb: results, close=lambda: None)

    monkeypatch.setattr(
        "hand_tracking.mediapipe_hands._create_native_hands",
        lambda **_kwargs: fake_hands,
    )
    tracker = MediaPipeHandsTracker()
    frame = np.zeros((80, 200, 3), dtype=np.uint8)
    hands = tracker.process(frame)

    assert tracker.available is True
    assert len(hands) == 1
    assert hands[0].handedness == "Left"
    assert len(hands[0].points) == 21
    assert hands[0].points[0] == pytest.approx((50.0, 40.0))
    tracker.close()


def _matplotlib_related(key: str) -> bool:
    return (
        key == "matplotlib"
        or key.startswith("matplotlib.")
        or key.startswith("mediapipe.python.solutions")
    )


class _MissingMatplotlibFinder:
    """模拟便携包排除 matplotlib，同时不破坏标准 import 协议。"""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "matplotlib" or fullname.startswith("matplotlib."):
            raise ModuleNotFoundError("No module named 'matplotlib'")
        return None


def _block_matplotlib_imports() -> tuple[dict, _MissingMatplotlibFinder]:
    saved = {key: sys.modules[key] for key in list(sys.modules) if _matplotlib_related(key)}
    for key in saved:
        del sys.modules[key]
    finder = _MissingMatplotlibFinder()
    sys.meta_path.insert(0, finder)
    return saved, finder


def _restore_blocked_modules(saved: dict, finder: _MissingMatplotlibFinder) -> None:
    if finder in sys.meta_path:
        sys.meta_path.remove(finder)
    for key in list(sys.modules):
        if key not in saved and _matplotlib_related(key):
            del sys.modules[key]
    sys.modules.update(saved)


def test_native_hands_init_survives_missing_matplotlib(monkeypatch):
    """Packaged exe excludes matplotlib; Hands init must not die on drawing_utils import."""
    pytest.importorskip("mediapipe")
    saved, finder = _block_matplotlib_imports()
    created = {}

    class DummyHands:
        def __init__(self, **kwargs):
            created.update(kwargs)

        def close(self):
            pass

    try:
        with pytest.raises(ImportError, match="matplotlib"):
            __import__("matplotlib")

        hands_mod._ensure_matplotlib_importable()
        import mediapipe.python.solutions.hands as mp_hands

        monkeypatch.setattr(mp_hands, "Hands", DummyHands)
        monkeypatch.setattr(hands_mod, "_path_is_ascii", lambda _path: True)

        instance = hands_mod._create_native_hands(max_num_hands=2)
        assert isinstance(instance, DummyHands)
        assert created["max_num_hands"] == 2

        tracker = MediaPipeHandsTracker()
        assert tracker.available is True
        assert "matplotlib" not in (tracker.error or "").lower()
        tracker.close()
    finally:
        _restore_blocked_modules(saved, finder)


def test_tracker_init_failure_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        "hand_tracking.mediapipe_hands._create_native_hands",
        lambda **_kwargs: (_ for _ in ()).throw(ImportError("no mediapipe")),
    )
    tracker = MediaPipeHandsTracker()
    frame = np.zeros((16, 16, 3), dtype=np.uint8)

    assert tracker.available is False
    with pytest.raises(RuntimeError, match="手部模型不可用"):
        tracker.process(frame)
    tracker.close()


def test_draw_hand_landmarks_paints_skeleton_pixels():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    painted = draw_hand_landmarks(
        frame,
        [{"handedness": "Right", "points": list(_points())}],
    )
    assert painted is not None
    blue = (painted[:, :, 0] >= 200) & (painted[:, :, 1] <= 40) & (painted[:, :, 2] <= 40)
    assert np.any(blue)


def test_worker_attaches_hand_landmarks_only_when_enabled():
    worker = CameraWorker(ConfigManager(), AppState())
    created: list[FakeTracker] = []

    def factory():
        tracker = FakeTracker()
        created.append(tracker)
        return tracker

    worker._hand_tracker_factory = factory
    frame = np.zeros((20, 30, 3), dtype=np.uint8)

    assert worker._hand_landmarks_payload(frame) is None
    assert created == []

    worker.set_hand_overlay_enabled(True)
    payload = worker._hand_landmarks_payload(frame)
    assert payload == landmarks_to_payload(created[0]._hands)
    assert len(created[0].process_calls) == 1

    worker.set_hand_overlay_enabled(False)
    assert worker._hand_landmarks_payload(frame) is None
    assert created[0].closed is True


def test_worker_factory_failure_does_not_raise():
    worker = CameraWorker(ConfigManager(), AppState())

    def factory():
        raise ImportError("no mediapipe")

    worker._hand_tracker_factory = factory
    worker.set_hand_overlay_enabled(True)
    frame = np.zeros((12, 12, 3), dtype=np.uint8)

    assert worker._hand_landmarks_payload(frame) == []
    assert worker._hand_landmarks_payload(frame) == []


def test_sequential_detect_keeps_hands_off_when_not_required_and_enables_only_for_hand_bundle(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = False
    config.action_order_constraint_enabled = False
    config.category_names = ["A", "", "", "", "", ""]
    config.category_counts = [1, 1, 1, 1, 1, 1]
    window = _window(config)
    window.state.set_camera_on(True)
    flags: list[bool] = []
    window.worker.set_hand_overlay_enabled = flags.append

    # 纯 YOLO 顺序模式不启用手部模型，避免白耗 CPU
    window._on_toggle_detect()
    assert window.state.inference_on is True
    assert flags[-1] is False

    # 加载无手部动作模型仍保持关闭
    window.worker.set_action_session(
        SimpleNamespace(bundle=SimpleNamespace(hands_in_training=False), reset=lambda: None)
    )
    window._sync_hand_overlay_enabled()
    assert flags[-1] is False

    # 加载带手部特征的动作模型后才按需开启
    window.worker.set_action_session(
        SimpleNamespace(bundle=SimpleNamespace(hands_in_training=True), reset=lambda: None)
    )
    window._sync_hand_overlay_enabled()
    assert flags[-1] is True

    # 停止检测立即关闭
    window._on_toggle_detect()
    assert window.state.inference_on is False
    assert flags[-1] is False


def test_roi_mode_does_not_enable_hand_overlay(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = True
    window = _window(config)
    window.state.set_inference_on(True)
    flags: list[bool] = []
    window.worker.set_hand_overlay_enabled = flags.append

    window._sync_hand_overlay_enabled()
    assert flags == [False]


def test_overlay_hands_do_not_enter_step_engine(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = False
    config.action_order_constraint_enabled = False
    window = _window(config)
    spy = EngineSpy()
    window.step_engine = spy
    points = list(_points())

    window._on_overlay_updated(
        {
            "source_frame_id": 9,
            "timestamp": 1.0,
            "model_path": "test.onnx",
            "task_type": "onnx_obb",
            "detections": [
                {
                    "class_id": 0,
                    "label": "A",
                    "conf": 0.9,
                    "track_id": None,
                    "polygon": [(10, 10), (30, 10), (30, 30), (10, 30)],
                    "box": (10, 10, 30, 30),
                    "center": (20, 20),
                }
            ],
            "image_size": (60, 40),
            "status": "",
            "error": "",
            "round_id": 0,
            "latency_ms": 5.0,
            "hand_landmarks": [{"handedness": "Right", "points": points}],
        }
    )

    assert len(spy.updates) == 1
    detections, frame_id = spy.updates[0]
    assert frame_id == 9
    assert [det.label for det in detections] == ["A"]
    assert window._latest_runtime_overlay.hand_landmarks[0]["handedness"] == "Right"
    assert len(window._latest_runtime_overlay.hand_landmarks[0]["points"]) == 21


def test_action_mode_uses_yolo_when_the_template_misses():
    from camera.camera_worker import CameraWorker
    from yolo_runtime.yolo_result_models import DetectionOverlayState

    class _Det:
        def __init__(self, label):
            self.label = label
            self.conf = 0.9
            self.box = (0, 0, 1, 1)

    class _Inner:
        def process_frame(self, frame, source_frame_id=0, round_id=0):
            return DetectionOverlayState(detections=[_Det("路由器")])

    class _Gated:
        def __init__(self):
            self._inner = _Inner()

        def process_frame(self, frame, source_frame_id=0, round_id=0):
            return DetectionOverlayState(detections=[], template_matched=False)

    worker = CameraWorker(ConfigManager(), AppState())
    worker.frame_processor = _Gated()
    worker.set_action_session(object())
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    overlay = DetectionOverlayState(detections=[], template_matched=False)
    found = worker._action_detections(frame, overlay, 3)
    assert [item.label for item in found] == ["路由器"]


def test_hands_only_overlay_is_drawn_while_detecting(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = False
    window = _window(config)
    window.state.set_camera_on(True)
    window.state.set_inference_on(True)
    overlay = DetectionOverlayState(
        source_frame_id=7,
        hand_landmarks=[{"handedness": "Left", "points": list(_points())}],
    )

    assert window._should_draw_runtime_overlay(overlay, 7) is True
    window.state.set_inference_on(False)
    assert window._should_draw_runtime_overlay(overlay, 7) is False
