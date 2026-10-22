from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtWidgets import QApplication

from app_state import AppState
from camera.camera_worker import CameraWorker
from config import ConfigManager
from main import (
    initialize_portable_config,
    load_yolo_processor_with_diagnostics,
    run_portable_smoke_check,
)
from runtime_diagnostics import (
    close_runtime_logger,
    configure_runtime_logger,
    get_runtime_logger,
    log_startup_context,
)
from realtime_diagnostics import run_realtime_diagnostics
from step_sequence.step_sequence_engine import StepSequenceEngine
from ui.main_window import MainWindow
from yolo_runtime.yolo_result_models import DetectionOverlayState


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def runtime_log(tmp_path: Path):
    log_path = configure_runtime_logger(tmp_path)
    try:
        yield log_path
    finally:
        close_runtime_logger()


def _read_log(path: Path) -> str:
    for handler in get_runtime_logger().handlers:
        handler.flush()
    return path.read_text(encoding="utf-8")


def test_latest_startup_log_replaces_previous_run(tmp_path: Path):
    first_path = configure_runtime_logger(tmp_path)
    get_runtime_logger().info("old-run-marker")
    close_runtime_logger()

    second_path = configure_runtime_logger(tmp_path)
    get_runtime_logger().info("current-run-marker")
    close_runtime_logger()

    assert first_path == second_path == tmp_path / "logs" / "startup_diagnostics.log"
    text = second_path.read_text(encoding="utf-8")
    assert "current-run-marker" in text
    assert "old-run-marker" not in text


def test_startup_context_and_processor_failure_include_actionable_details(
    runtime_log: Path,
    tmp_path: Path,
):
    config_path = tmp_path / "config.json"
    log_startup_context(
        base_dir=tmp_path,
        resource_dir=tmp_path / "_internal",
        config_path=config_path,
        frozen=True,
    )
    config = ConfigManager()
    config.yolo_model_path = "config/missing.onnx"

    processor = load_yolo_processor_with_diagnostics(config, app_base_dir=tmp_path)

    assert processor is None
    text = _read_log(runtime_log)
    assert "frozen=True" in text
    assert f"config_path={config_path}" in text
    assert f"model_path={tmp_path / 'config' / 'missing.onnx'}" in text
    assert "model_exists=False" in text
    assert "processor_load=failed" in text
    assert "FileNotFoundError" in text
    assert "Traceback (most recent call last)" in text


def test_portable_config_source_is_recorded(runtime_log: Path, tmp_path: Path):
    app_dir = tmp_path / "app"
    resource_dir = tmp_path / "_internal"
    app_dir.mkdir()
    resource_dir.mkdir()
    bundled_config = resource_dir / "config.json"
    bundled_config.write_text('{"yolo_model_path": "E:/old/model.onnx"}', encoding="utf-8")

    config_path = initialize_portable_config(app_dir, resource_dir)

    assert config_path.read_text(encoding="utf-8") == bundled_config.read_text(encoding="utf-8")
    text = _read_log(runtime_log)
    assert "portable_config=created" in text
    assert f"source={bundled_config}" in text
    assert f"destination={config_path}" in text


def test_detection_rejection_shows_copyable_log_path(
    qapp,
    runtime_log: Path,
    tmp_path: Path,
):
    config = ConfigManager()
    state = AppState()
    state.set_camera_on(True)
    window = MainWindow(
        config,
        state,
        processor=None,
        step_engine=StepSequenceEngine(config.category_names),
        app_base_dir=tmp_path,
        runtime_log_path=runtime_log,
    )

    window._on_toggle_detect()

    assert "处理器未加载" in window._status_label.text()
    assert str(runtime_log) in window._notice_label.text()
    assert "detection_start=rejected reason=processor_unavailable" in _read_log(runtime_log)
    window.close()


def test_inference_overlay_error_is_written_once(runtime_log: Path):
    worker = CameraWorker(ConfigManager(), AppState())
    overlay = DetectionOverlayState(
        source_frame_id=9,
        model_path="config/model.onnx",
        error="推理失败: DLL load failed",
    )

    worker._record_inference_diagnostics(overlay)
    worker._record_inference_diagnostics(overlay)

    text = _read_log(runtime_log)
    assert text.count("inference_result=failed") == 1
    assert "frame_id=9" in text
    assert "DLL load failed" in text


def test_portable_smoke_attempts_real_processor_load(runtime_log: Path, tmp_path: Path):
    config = ConfigManager()
    config.yolo_model_path = "config/model.onnx"
    config_path = tmp_path / "config.json"
    config.save(config_path)
    model_path = tmp_path / "config" / "model.onnx"
    model_path.parent.mkdir()
    model_path.write_bytes(b"test-model")
    calls = []

    def failed_loader(runtime_config, app_base_dir):
        calls.append((runtime_config.yolo_model_path, Path(app_base_dir)))
        get_runtime_logger().error("processor_load=failed simulated")
        return None

    code = run_portable_smoke_check(
        config_path=config_path,
        app_base_dir=tmp_path,
        resource_dir=tmp_path,
        processor_loader=failed_loader,
    )

    assert code == 6
    assert calls == [("config/model.onnx", tmp_path)]
    smoke_text = (tmp_path / "logs" / "portable_smoke_test.log").read_text(encoding="utf-8")
    assert "processor load failed" in smoke_text
    assert str(runtime_log) in smoke_text


def test_portable_smoke_rejects_plain_processor_when_template_gate_is_enabled(
    runtime_log: Path,
    tmp_path: Path,
):
    class PlainProcessor:
        def get_class_names(self):
            return {0: "A"}

    config = ConfigManager()
    config.yolo_model_path = "config/model.onnx"
    config.category_names = ["A"]
    config.category_counts = [1]
    config.roi_inspection_enabled = True
    config.roi_inspection_rect = {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8}
    config.roi_template_enabled = True
    config.roi_template_rect = {"x": 0.2, "y": 0.2, "width": 0.2, "height": 0.2}
    config.roi_template_search_rect = {
        "x": 0.1,
        "y": 0.1,
        "width": 0.6,
        "height": 0.6,
    }
    config.roi_template_path = "locator_template.png"
    (tmp_path / "locator_template.png").write_bytes(b"template")
    config_path = tmp_path / "config.json"
    config.save(config_path)
    model_path = tmp_path / "config" / "model.onnx"
    model_path.parent.mkdir()
    model_path.write_bytes(b"test-model")

    code = run_portable_smoke_check(
        config_path=config_path,
        app_base_dir=tmp_path,
        resource_dir=tmp_path,
        processor_loader=lambda *_args: PlainProcessor(),
    )

    assert code == 6
    smoke_text = (tmp_path / "logs" / "portable_smoke_test.log").read_text(
        encoding="utf-8"
    )
    assert "template gate processor not loaded" in smoke_text


def test_portable_and_realtime_diagnostics_reject_legacy_config_before_runtime(tmp_path: Path):
    config_path = tmp_path / "legacy.json"
    config_path.write_text(
        '{"model_task":"ultralytics_obb","yolo_model_path":"config/model.onnx",'
        '"category_names":["A"],"category_counts":[1]}',
        encoding="utf-8",
    )

    smoke_code = run_portable_smoke_check(
        config_path=config_path,
        app_base_dir=tmp_path,
        resource_dir=tmp_path,
        processor_loader=lambda *_args: pytest.fail("不应加载处理器"),
    )
    realtime = run_realtime_diagnostics(
        config_path=config_path,
        duration_s=0.1,
        processor_factory=lambda *_args: pytest.fail("不应加载处理器"),
    )

    assert smoke_code == 1
    assert realtime["error"] == "inspection_config_invalid"
    assert any("配置版本" in error for error in realtime["config_errors"])


def test_windowed_runtime_hook_restores_missing_standard_streams(
    monkeypatch,
):
    hook_path = Path(__file__).resolve().parents[1] / "packaging" / "windowed_stdio.py"
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    runpy.run_path(str(hook_path))

    assert sys.stdout is not None
    assert sys.stderr is not None
    assert sys.stdout.write("stdout probe") == len("stdout probe")
    assert sys.stderr.write("stderr probe") == len("stderr probe")
