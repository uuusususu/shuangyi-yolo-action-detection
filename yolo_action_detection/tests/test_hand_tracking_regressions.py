"""手部故障不能伪装成无手，产线启动不依赖工作台预热。"""
from pathlib import Path
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "yolo_action_detection" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hand_tracking import mediapipe_hands as hands_module
from lstm_action.bundle import _check_hands, load_runtime_bundle
from lstm_workbench import model_registry


def test_native_failure_is_observable_and_raises(monkeypatch):
    def broken(_frame):
        raise ValueError("graph failure")
    monkeypatch.setattr(hands_module, "_create_native_hands", lambda **_kw: SimpleNamespace(process=broken, close=lambda: None))
    tracker = hands_module.MediaPipeHandsTracker()
    with pytest.raises(RuntimeError, match="graph failure"):
        tracker.process(np.zeros((10, 10, 3), np.uint8))
    assert not tracker.available
    assert "graph failure" in tracker.error
    with pytest.raises(RuntimeError, match="graph failure"):
        tracker.process(np.zeros((10, 10, 3), np.uint8))


def test_no_visible_hands_remains_valid_empty_result(monkeypatch):
    monkeypatch.setattr(hands_module, "_create_native_hands", lambda **_kw: SimpleNamespace(process=lambda _f: SimpleNamespace(), close=lambda: None))
    tracker = hands_module.MediaPipeHandsTracker()
    assert tracker.process(np.zeros((10, 10, 3), np.uint8)) == []
    assert tracker.available and not tracker.error
    tracker.close()


def test_bundle_probes_in_a_cold_process_without_warmup(monkeypatch):
    monkeypatch.setattr(model_registry, "hands_status", lambda: {"state": "idle"})
    probed = []
    monkeypatch.setattr(model_registry, "probe_hands", lambda: probed.append(True) or {"ok": True, "sha256": "actual", "model_complexity": 0})
    _check_hands({"hands_sha256": "actual", "hands_model_complexity": 0}, None, None)
    assert probed == [True]


def test_bundle_checks_actual_complexity_even_when_only_sha_was_supplied(monkeypatch):
    monkeypatch.setattr(model_registry, "probe_hands", lambda: {"ok": True, "sha256": "actual", "model_complexity": 0})
    with pytest.raises(ValueError, match="复杂度"):
        _check_hands({"hands_sha256": "actual", "hands_model_complexity": 1}, "actual", None)


def test_probe_closes_its_own_session(monkeypatch):
    closed = []
    monkeypatch.setattr(model_registry, "hands_fingerprint", lambda: {"ok": True, "sha256": "actual", "model_complexity": 0})
    monkeypatch.setattr(model_registry, "create_hands_tracker", lambda: SimpleNamespace(available=True, close=lambda: closed.append(True)), raising=False)
    assert model_registry.probe_hands()["ok"]
    assert closed == [True]


def test_live_backends_owns_independent_sessions(monkeypatch):
    sessions = []
    class Tracker:
        available = True
        error = ""
        def __init__(self):
            self.frames = 0
            self.closed = False
            sessions.append(self)
        def process(self, _f):
            self.frames += 1
            return []
        def close(self):
            self.closed = True
    monkeypatch.setattr(model_registry, "create_hands_tracker", Tracker, raising=False)
    monkeypatch.setattr(model_registry, "load_yolo", lambda *_a: SimpleNamespace(process_frame=lambda _f: SimpleNamespace(error="", detections=[])))
    first_detect, first_hands = model_registry.live_backends({"yolo_path": "fake"})
    second_detect, second_hands = model_registry.live_backends({"yolo_path": "fake"})
    first_hands(None)
    first_hands(None)
    second_hands(None)
    first_hands.close()
    second_hands.close()
    assert [t.frames for t in sessions] == [2, 1]
    assert all(t.closed for t in sessions)


def test_resource_mirror_detects_same_size_same_timestamp_replacement(tmp_path):
    source = tmp_path / "source" / "hand_landmark"
    source.mkdir(parents=True)
    model = source / "model.tflite"
    model.write_bytes(b"first")
    timestamp = model.stat().st_mtime_ns
    first = hands_module._module_signature(source)
    model.write_bytes(b"other")
    os.utime(model, ns=(timestamp, timestamp))
    assert hands_module._module_signature(source) != first


def test_runtime_bundle_rejects_nonexistent_configured_yolo(tmp_path):
    from yolo_action_detection.tests.test_lstm_sequence_runtime import _bundle
    directory, _yolo = _bundle(tmp_path)
    with pytest.raises(ValueError, match="YOLO.*不存在"):
        load_runtime_bundle(directory, category_names=["包装", "封装"], yolo_path=str(tmp_path / "missing.onnx"), hands_sha="hands", hands_complexity=0)
