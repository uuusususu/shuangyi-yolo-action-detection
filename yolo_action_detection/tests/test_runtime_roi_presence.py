"""检测时按有无加载识别 ROI / 模板门控。"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtWidgets import QApplication

from app_state import AppState
from config import ConfigManager
from roi_inspection.models import InspectionOutcome, RoundPhase
from step_sequence.step_sequence_engine import StepSequenceEngine, StepStatus
from ui.main_window import MainWindow


class DummyProcessor:
    def get_class_names(self):
        return {0: "A", 1: "B"}


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def _base_config(tmp_path: Path) -> ConfigManager:
    config = ConfigManager()
    config._config_path = str(tmp_path / "config.json")
    config.category_names = ["A", "B", "", "", "", ""]
    config.category_counts = [1, 1, 1, 1, 1, 1]
    config.action_pass_stable_frames = 1
    config.action_ng_stable_frames = 2
    config.round_cooldown_seconds = 60.0
    return config


def _window(qapp, config) -> MainWindow:
    return MainWindow(
        config,
        AppState(),
        processor=DummyProcessor(),
        step_engine=StepSequenceEngine(
            step_class_names=config.category_names,
            step_counts=config.get_runtime_category_counts(),
        ),
        processor_factory=lambda *args, **kwargs: DummyProcessor(),
        app_base_dir=Path(config.get_config_path()).parent,
    )


def _det(label, x, y):
    return {
        "class_id": 0,
        "label": label,
        "conf": 0.9,
        "track_id": None,
        "polygon": [[x - 5, y - 5], [x + 5, y - 5], [x + 5, y + 5], [x - 5, y + 5]],
        "box": [x - 5, y - 5, x + 5, y + 5],
        "center": [x, y],
    }


def _payload(detections, frame_id=1):
    return {
        "source_frame_id": frame_id,
        "timestamp": time.monotonic(),
        "model_path": "model.onnx",
        "task_type": "ultralytics_obb",
        "detections": detections,
        "image_size": (1000, 1000),
        "source_frame": np.full((1000, 1000, 3), frame_id % 255, dtype=np.uint8),
    }


def _spy_step_update(window):
    received = []
    original = window.step_engine.update

    def spy(detections, *args, **kwargs):
        received.append(list(detections))
        return original(detections)

    window.step_engine.update = spy
    return received


def test_sequential_inspection_roi_filters_outside_centers(qapp, tmp_path):
    config = _base_config(tmp_path)
    config.roi_inspection_enabled = False
    config.action_order_constraint_enabled = False
    config.roi_inspection_rect = {"x": 0.5, "y": 0.0, "width": 0.5, "height": 1.0}
    window = _window(qapp, config)
    received = _spy_step_update(window)
    window.step_engine.start_round(require_first_step_rearm=False)
    outside = _det("A", 100, 100)
    inside = _det("A", 600, 100)

    window._on_overlay_updated(_payload([outside, inside]))

    assert [tuple(det.center) for det in received[0]] == [(600, 100)]
    assert window.step_engine.get_state().steps[0].status == StepStatus.PASS
    overlay_centers = [tuple(det.center) for det in window._latest_runtime_overlay.detections]
    assert (100, 100) in overlay_centers
    assert (600, 100) in overlay_centers


def test_sequential_empty_or_invalid_inspection_uses_all_detections(qapp, tmp_path):
    config = _base_config(tmp_path)
    config.roi_inspection_enabled = False
    config.action_order_constraint_enabled = False
    config.roi_inspection_rect = {"x": -0.2, "y": 0.0, "width": 0.5, "height": 1.0}
    window = _window(qapp, config)
    received = _spy_step_update(window)
    window.step_engine.start_round(require_first_step_rearm=False)

    window._on_overlay_updated(_payload([_det("A", 100, 100)]))

    assert [tuple(det.center) for det in received[0]] == [(100, 100)]
    assert window.step_engine.get_state().steps[0].status == StepStatus.PASS


def test_sequential_empty_inspection_allows_start(qapp, tmp_path, monkeypatch):
    config = _base_config(tmp_path)
    config.roi_inspection_enabled = False
    config.action_order_constraint_enabled = False
    config.roi_inspection_rect = None
    window = _window(qapp, config)
    window.state.camera_on = True
    calls = []
    monkeypatch.setattr(window.worker, "start_inference", lambda: calls.append(True))

    window._on_toggle_detect()

    assert calls == [True]
    assert "识别 ROI 未配置" not in window._notice_label.text()


def test_unordered_empty_inspection_counts_full_frame(qapp, tmp_path):
    config = _base_config(tmp_path)
    config.roi_inspection_enabled = True
    config.roi_inspection_rect = None
    window = _window(qapp, config)

    window._on_overlay_updated(_payload([_det("A", 100, 100), _det("B", 200, 100)]))

    assert window._roi_engine is not None
    assert window._roi_engine.snapshot.phase == RoundPhase.RESULT_HOLD
    assert window._roi_engine.snapshot.result.outcome == InspectionOutcome.PASS
    assert window._stats_manager.batch.ok == 1


def test_unordered_empty_inspection_allows_start(qapp, tmp_path, monkeypatch):
    config = _base_config(tmp_path)
    config.roi_inspection_enabled = True
    config.roi_inspection_rect = None
    window = _window(qapp, config)
    window.state.camera_on = True
    calls = []
    monkeypatch.setattr(window.worker, "start_inference", lambda: calls.append(True))

    window._on_toggle_detect()

    assert calls == [True]
    assert window._roi_engine is not None
    assert "识别 ROI 未配置" not in window._notice_label.text()


def test_template_switch_on_but_missing_file_does_not_wait_for_locate(qapp, tmp_path):
    config = _base_config(tmp_path)
    config.roi_inspection_enabled = True
    config.roi_inspection_rect = {"x": 0.5, "y": 0.0, "width": 0.5, "height": 1.0}
    config.roi_template_enabled = True
    config.roi_template_path = "locator_template.png"
    config.roi_template_search_rect = {"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.3}
    window = _window(qapp, config)

    assert window._roi_engine is not None
    assert window._roi_engine.snapshot.phase == RoundPhase.WAITING_FOR_LOCATOR
    assert window._roi_engine._config.template_enabled is False

    window._on_overlay_updated(_payload([_det("A", 600, 100), _det("B", 700, 100)]))

    assert window._roi_engine.snapshot.phase == RoundPhase.RESULT_HOLD
    assert window._roi_engine.snapshot.result.outcome == InspectionOutcome.PASS


def test_incomplete_template_layer_allows_start_without_unconfigured_notice(
    qapp,
    tmp_path,
    monkeypatch,
):
    config = _base_config(tmp_path)
    config.roi_inspection_enabled = False
    config.action_order_constraint_enabled = False
    config.roi_template_enabled = True
    config.roi_template_path = "locator_template.png"
    config.roi_template_search_rect = {"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.3}
    window = _window(qapp, config)
    window.state.camera_on = True
    calls = []
    monkeypatch.setattr(window.worker, "start_inference", lambda: calls.append(True))

    window._on_toggle_detect()

    assert calls == [True]
    notice = window._notice_label.text()
    assert "搜索 ROI 未配置" not in notice
    assert "识别 ROI 未配置" not in notice
    assert "模板定位处理器未加载" not in notice


def test_overlay_stops_drawing_search_roi_after_it_is_cleared(qapp, tmp_path):
    config = _base_config(tmp_path)
    config.roi_template_search_rect = {"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.3}
    window = _window(qapp, config)
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    assert tuple(int(value) for value in window._draw_roi_overlay(frame.copy())[100, 100]) != (0, 0, 0)

    config.roi_template_search_rect = None
    rendered = window._draw_roi_overlay(np.zeros((1000, 1000, 3), dtype=np.uint8))

    assert tuple(int(value) for value in rendered[100, 100]) == (0, 0, 0)


def test_sequential_overlay_draws_saved_inspection_roi(qapp, tmp_path):
    config = _base_config(tmp_path)
    config.roi_inspection_enabled = False
    config.roi_inspection_rect = {"x": 0.5, "y": 0.0, "width": 0.5, "height": 1.0}
    window = _window(qapp, config)
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)

    rendered = window._draw_roi_overlay(frame)

    assert tuple(int(value) for value in rendered[0, 500]) != (0, 0, 0)
    assert tuple(int(value) for value in rendered[100, 100]) == (0, 0, 0)
