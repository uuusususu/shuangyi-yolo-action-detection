from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app_state import AppState
from camera.camera_worker import CameraWorker
from yolo_runtime.yolo_result_models import DetectionOverlayState


class Session:
    def __init__(self):
        self.fault = ""
        self.observed = []

    def set_fault(self, reason):
        self.fault = str(reason)
        return {"fault": self.fault, "events": [], "phase": "故障"}

    def observe_frame(self, **kwargs):
        self.observed.append(kwargs)
        return {"fault": "", "events": [], "evidence_frames": []}

    def reset(self):
        self.fault = ""


def worker():
    return CameraWorker(SimpleNamespace(), AppState())


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_overlay_error_locks_action_session_instead_of_becoming_empty_detection():
    camera = worker()
    session = Session()
    camera.set_action_session(session)
    frame = np.zeros((4, 6, 3), np.uint8)
    report = camera._observe_action(frame, 8, 1.0, DetectionOverlayState(error="YOLO failed"), [])
    assert report["fault"] == "YOLO failed"
    assert session.fault == "YOLO failed"
    assert session.observed == []


@pytest.mark.parametrize("returned_error", [False, True])
def test_template_fallback_yolo_failure_is_not_empty_success(returned_error):
    camera = worker()
    camera.set_action_session(Session())

    def process(*args, **kwargs):
        if returned_error:
            return DetectionOverlayState(error="backend failed")
        raise ValueError("backend failed")

    camera.frame_processor = SimpleNamespace(_inner=SimpleNamespace(process_frame=process))
    with pytest.raises(RuntimeError, match="backend failed"):
        camera._action_detections(np.zeros((4, 6, 3), np.uint8), DetectionOverlayState(), 8)


def test_tracker_returned_error_is_fault_but_normal_no_hands_is_valid():
    camera = worker()
    tracker = SimpleNamespace(available=True, error="", process=lambda frame: [])
    camera._hand_tracker_factory = lambda: tracker
    frame = np.zeros((4, 6, 3), np.uint8)
    assert camera._hand_landmarks_payload(frame, force=True) == []
    assert camera._hand_runtime_error == ""
    tracker.error = "graph failed"
    assert camera._hand_landmarks_payload(frame, force=True) == []
    assert camera._hand_runtime_error == "graph failed"


def test_evidence_is_exact_owned_bounded_and_cleared_by_hard_reset():
    camera = worker()
    camera.set_action_session(Session())
    for frame_id in range(1, 500):
        frame = np.full((4, 6, 3), frame_id % 255, np.uint8)
        camera._cache_action_evidence(frame, frame_id, float(frame_id), [], "model")
        camera._prune_action_evidence({"evidence_frames": [1, frame_id]})
        frame[:] = 0
    evidence = camera.get_action_boundary_evidence(1)
    assert evidence is not None
    image, detections, metadata = evidence
    assert np.all(image == 1)
    assert metadata["source_frame_id"] == 1
    image[:] = 0
    assert np.all(camera.get_action_boundary_evidence(1)[0] == 1)
    assert camera.get_action_boundary_evidence(42) is None
    assert len(camera._action_evidence) <= camera._ACTION_EVIDENCE_LIMIT
    camera.reset_action_session()
    assert camera.get_action_boundary_evidence(1) is None


def test_exclusive_boundary_evidence_uses_successor_pixels_and_detections():
    camera = worker()
    camera.set_action_session(Session())
    for frame_id, label in [(31, "old"), (38, "successor"), (45, "confirmation")]:
        frame = np.full((4, 6, 3), frame_id, np.uint8)
        camera._cache_action_evidence(frame, frame_id, float(frame_id), [{"label": label}], "model", wall_time=1792051200.0 + frame_id)
    camera._prune_action_evidence({"events": [{"start_frame": 31, "end_frame": 38, "decision_frame": 45}]})
    image, detections, metadata = camera.get_action_boundary_evidence(38)
    assert np.all(image == 38)
    assert detections == [{"label": "successor"}]
    assert metadata["source_frame_id"] == 38
    assert metadata["capture_monotonic"] == 38.0
    assert metadata["timestamp"] == 1792051238.0
    assert camera.get_action_boundary_evidence(37) is None
    for frame_id in range(46, 52):
        camera._cache_action_evidence(np.zeros((4, 6, 3), np.uint8), frame_id, float(frame_id), [], "model")
        camera._prune_action_evidence({"evidence_frames": [frame_id]})
    # Qt 界面延迟取证时仍保留最近动作真实边界。
    assert np.all(camera.get_action_boundary_evidence(38)[0] == 38)


def test_runtime_swap_is_atomic_while_stopped_and_rejects_active_inference():
    camera = worker()
    old = (camera.config, object(), Session())
    camera.replace_runtime(*old)
    assert camera.get_action_session() is old[2]
    camera.state.set_camera_on(True)
    camera.state.set_inference_on(True)
    new = (SimpleNamespace(), object(), Session())
    with pytest.raises(RuntimeError):
        camera.replace_runtime(*new)
    assert (camera.config, camera.frame_processor, camera.get_action_session()) == old
    assert camera.runtime_generation == 1
    camera.state.set_inference_on(False)
    camera.replace_runtime(*new)
    assert (camera.config, camera.frame_processor, camera.get_action_session()) == new
    assert camera.runtime_generation == 2


def test_hard_reset_and_session_replacement_advance_overlay_generation():
    camera = worker()
    before = camera.runtime_generation
    camera.set_action_session(Session())
    assert camera.runtime_generation > before
    before = camera.runtime_generation
    camera.reset_action_session()
    assert camera.runtime_generation > before
    before = camera.runtime_generation
    camera._clear_session_buffers()
    assert camera.runtime_generation > before


@pytest.mark.parametrize("processor_fails", [False, True])
def test_loop_fault_is_reported_and_display_uses_actual_input_frame(processor_fails, qapp):
    camera = worker()
    session = Session()
    camera.set_action_session(session)
    camera._hand_tracker_factory = lambda: SimpleNamespace(available=True, error="", process=lambda frame: [])
    camera.running = True
    camera.state.set_camera_on(True)
    camera.state.set_inference_on(True)
    camera._latest_infer_frame = np.full((4, 6, 3), 17, np.uint8)
    camera._latest_infer_frame_id = 17
    camera._latest_infer_captured_at = 1.7
    camera.error_occurred.connect(lambda message: setattr(camera, "running", False))

    def process(*args, **kwargs):
        if processor_fails:
            raise ValueError("processing failed")
        return DetectionOverlayState(source_frame_id=0, error="YOLO failed")

    camera.frame_processor = SimpleNamespace(process_frame=process)
    emitted = []

    def on_overlay(payload):
        emitted.append(payload)
        camera.running = False

    camera.overlay_state_updated.connect(on_overlay)
    camera._inference_loop(camera._session_generation)
    assert len(emitted) == 1
    assert emitted[0]["lstm_action"]["fault"]
    assert emitted[0]["source_frame_id"] == camera.get_display_frame_id() == 17
    assert np.all(emitted[0]["source_frame"] == 17)
    assert session.fault
