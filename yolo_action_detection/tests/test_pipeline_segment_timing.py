"""分段耗时统计。

"框追不上目标"可能卡在采集、模板定位、YOLO 推理或界面渲染任何一段，
只报一个总延迟无法定位，所以四段都要单独可读。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtWidgets import QApplication

from app_state import AppState
from camera.camera_worker import CameraWorker
from config import ConfigManager
from step_sequence.step_sequence_engine import StepSequenceEngine
from template_matching.shape_model import ShapeMatchResult
from ui.main_window import MainWindow
from yolo_runtime.template_gated_processor import TemplateGatedProcessor
from yolo_runtime.yolo_result_models import DetectionOverlayState, PipelineStats


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


class _LocatorStub:
    def __init__(self, result: ShapeMatchResult) -> None:
        self.result = result

    def find(self, frame, search_rect, *, collect_angle_curve=True):
        return self.result


class _InnerStub:
    def __init__(self, latency_ms: float) -> None:
        self._latency_ms = latency_ms

    def get_class_names(self):
        return {0: "A"}

    def process_frame(self, frame, source_frame_id=0, round_id=0):
        return DetectionOverlayState(
            source_frame_id=source_frame_id,
            round_id=round_id,
            latency_ms=self._latency_ms,
            yolo_ms=self._latency_ms,
        )


def test_pipeline_stats_expose_all_four_segments():
    stats = PipelineStats(
        capture_ms=3.0, template_ms=26.0, yolo_ms=41.0, render_ms=8.0
    )

    payload = stats.to_dict()

    assert payload["capture_ms"] == 3.0
    assert payload["template_ms"] == 26.0
    assert payload["yolo_ms"] == 41.0
    assert payload["render_ms"] == 8.0


def test_gated_processor_splits_template_and_yolo_time():
    locator = _LocatorStub(
        ShapeMatchResult(matched=True, score=0.9, center=(10.0, 10.0), angle=0.0,
                         total_ms=26.0))
    processor = TemplateGatedProcessor(
        _InnerStub(latency_ms=41.0), locator,
        {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5})

    overlay = processor.process_frame(np.zeros((40, 60, 3), np.uint8), source_frame_id=1)

    assert overlay.template_ms == 26.0
    assert overlay.yolo_ms == 41.0
    # 总延迟必须覆盖两段，否则现场看到的"推理耗时"会漏掉定位那一半
    assert overlay.latency_ms > 0.0


def test_gated_processor_reports_template_time_when_not_matched():
    locator = _LocatorStub(ShapeMatchResult(matched=False, score=0.2, total_ms=26.0))
    processor = TemplateGatedProcessor(
        _InnerStub(latency_ms=41.0), locator,
        {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5})

    overlay = processor.process_frame(np.zeros((40, 60, 3), np.uint8), source_frame_id=2)

    assert overlay.template_ms == 26.0
    assert overlay.yolo_ms == 0.0


def test_worker_records_render_latency_reported_by_ui():
    worker = CameraWorker(ConfigManager(), AppState())

    worker.record_render_latency(12.5)

    assert worker.get_pipeline_stats()["render_ms"] == 12.5


def _timing_window(config: ConfigManager) -> MainWindow:
    return MainWindow(
        config,
        AppState(),
        processor=None,
        step_engine=StepSequenceEngine(
            step_class_names=config.category_names,
            step_counts=config.get_runtime_category_counts(),
        ),
    )


def test_timing_readout_names_the_slowest_segment(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = False
    window = _timing_window(config)
    window.state.set_camera_on(True)
    window.state.set_inference_on(True)

    window._on_stats_updated({
        "capture_ms": 3.0,
        "template_ms": 26.0,
        "yolo_ms": 41.0,
        "render_ms": 8.0,
        "infer_fps": 12.5,
    })

    text = window._timing_label.text()
    assert window._timing_label.isHidden() is False
    assert "采集 3.0ms" in text
    assert "定位 26.0ms" in text
    assert "推理 41.0ms" in text
    assert "渲染 8.0ms" in text
    assert "瓶颈 推理" in text
    assert "12.5 fps" in text


def test_timing_readout_omits_segments_that_did_not_run(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = False
    window = _timing_window(config)
    window.state.set_camera_on(True)
    window.state.set_inference_on(True)

    window._on_stats_updated({
        "capture_ms": 3.0,
        "template_ms": 0.0,
        "yolo_ms": 41.0,
        "render_ms": 8.0,
        "infer_fps": 20.0,
    })

    text = window._timing_label.text()
    assert "定位" not in text
    assert "瓶颈 推理" in text


def test_timing_readout_hidden_when_detection_is_off(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = False
    window = _timing_window(config)
    window.state.set_camera_on(True)
    window.state.set_inference_on(True)
    window._on_stats_updated({"capture_ms": 3.0, "yolo_ms": 41.0, "infer_fps": 20.0})
    assert window._timing_label.text()

    window.state.set_inference_on(False)
    window._update_timing_display()

    assert window._timing_label.isHidden() is True


def test_display_update_reports_render_latency_to_worker(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = False
    window = _timing_window(config)
    window.state.set_camera_on(True)
    preview = np.zeros((40, 60, 3), dtype=np.uint8)
    window.worker.get_latest_preview_frame = lambda: preview.copy()
    window.worker.get_latest_preview_frame_id = lambda: 4

    window._update_display()

    assert window.worker.get_pipeline_stats()["render_ms"] > 0.0
